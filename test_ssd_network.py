# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a SSD model
on a given dataset."""
import math
import sys
import six
import time

import numpy as np
import tensorflow as tf
import tf_extended as tfe
import tf_utils
from tensorflow.python.framework import ops
import util
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import ssd_vgg_preprocessing
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'
# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'keep_threshold', 0.0, 'Keep detected objects with confidence not less than it')
    
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.1, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'matching_threshold', 0.9, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer(
    'eval_resize', 4, 'Image resizing: None / CENTRAL_CROP / PAD_AND_RESIZE / WARP_RESIZE.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size.')
tf.app.flags.DEFINE_boolean(
    'remove_difficult', True, 'Remove difficult objects from evaluation.')

# =========================================================================== #
# Main evaluation flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', None, 'Directory where the results are saved to.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'icdar2013', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge2.Task123/'), 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', None, 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_boolean(
    'wait_for_checkpoints', True, 'Wait for new checkpoints in the eval loop.')


FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        # =================================================================== #
        # SSD model + Pre-processing
        # =================================================================== #

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)

        # Evaluation shape and associated anchors: eval_image_size
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
#        image_preprocessing_fn = ssd_vgg_preprocessing.preprocess_for_test
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)


        with tf.device('/cpu:0'):
            image = tf.placeholder("float32", name = 'images', shape = [None, None, 3])
            gbboxes = tf.placeholder("float32", name = 'bboxes', shape = [None, 4])
            glabels = tf.placeholder('int32', name = 'labels', shape = [None, 1])

            # Pre-processing image, labels and bboxes.
            image_processed, glabels, gbboxes, gbbox_img = \
                image_preprocessing_fn(image, glabels, gbboxes,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT,
                                       resize=FLAGS.eval_resize,
                                       difficults=None)
            image_processed = tf.expand_dims(image_processed, 0)

        # =================================================================== #
        # SSD Network + Ouputs decoding.
        # =================================================================== #
        arg_scope = ssd_net.arg_scope(data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = \
                ssd_net.net(image_processed, is_training=False)

        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
            # Detected objects from SSD output.
            localisations = ssd_net.bboxes_decode(localisations, ssd_anchors)
            rscores, rbboxes = ssd_net.detected_bboxes(predictions, localisations,
                                        select_threshold=FLAGS.select_threshold,
                                        nms_threshold=FLAGS.nms_threshold,
                                        clipping_bbox=None,
                                        top_k=FLAGS.select_top_k,
                                        keep_top_k=FLAGS.keep_top_k)
            
        # =================================================================== #
        # Evaluation loop.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        
        if util.io.is_dir(FLAGS.checkpoint_path):
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            ckpt_path = ckpt.model_checkpoint_path
        else:
            ckpt_path = FLAGS.checkpoint_path;
        saver = tf.train.Saver()

        def check(index, score, score_threshold = FLAGS.keep_threshold, num_threshold = FLAGS.keep_top_k):
          if score < score_threshold:
            return False
          if index > num_threshold:
            return False
            
          return True
          
        def write_result(image_name, image_data, bboxes, scores, path):
          filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
          bboxes = bboxes.copy()
          print filename
          h, w = image_data.shape[0:-1]
          bboxes[:, 0] = bboxes[:, 0] * h
          bboxes[:, 1] = bboxes[:, 1] * w
          bboxes[:, 2] = bboxes[:, 2] * h
          bboxes[:, 3] = bboxes[:, 3] * w
          lines = []
          for b_idx, bscore in enumerate(scores):
                if not check(b_idx, bscore):
                  break
                bbox = bboxes[b_idx, :]
                [ymin, xmin, ymax, xmax] = [int(v) for v in bbox]
                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(w - 1, xmax)
                ymax = min(h - 1, ymax)
                line = "%d, %d, %d, %d\r\n"%(xmin, ymin, xmax, ymax)
                lines.append(line)
          util.io.write_lines(filename, lines)
          
        
        def draw_bbox(image_data, bboxes, scores, color):
          bboxes = bboxes.copy()
          image_data = image_data.copy()
          h, w = image_data.shape[0:-1]
          bboxes[:, 0] = bboxes[:, 0] * h
          bboxes[:, 1] = bboxes[:, 1] * w
          bboxes[:, 2] = bboxes[:, 2] * h
          bboxes[:, 3] = bboxes[:, 3] * w
          
          for b_idx, bscore in enumerate(scores):
                if not check(b_idx, bscore):
                  break;
                bbox = bboxes[b_idx, :]
                [ymin, xmin, ymax, xmax] = [int(v) for v in bbox]
                util.img.rectangle(image_data, (xmin, ymin), (xmax, ymax), color = color, border_width = 1)
                aspect_ratio = (xmax - xmin) * 1.0 / (ymax - ymin)
                util.img.put_text(image_data, '%.3f, %.2f'%(bscore, aspect_ratio), (xmin, ymin), scale = 0.7, color=color);
          return image_data
        
          
        with tf.Session() as sess:
            step = saver.restore(sess, ckpt_path)
            import datasets.icdar2013_data
            data_provider = datasets.icdar2013_data.ICDAR2013Data(root_dir = FLAGS.dataset_dir, split = FLAGS.dataset_split_name)
            ckpt_name = util.io.get_filename(str(ckpt_path));
            ckpt_dir = util.io.get_dir(str(ckpt_path));
            dump_path = util.io.join_path(ckpt_dir, 'eval')
            xml_path = util.io.join_path(dump_path, ckpt_name, FLAGS.dataset_split_name,  'xml')
            txt_path = util.io.join_path(dump_path, ckpt_name, FLAGS.dataset_split_name, 'txt')
            zip_path = util.io.join_path(dump_path, ckpt_name, FLAGS.dataset_split_name, ckpt_name + '.' + FLAGS.dataset_split_name +'.zip')
            def create_zip():
              cmd = 'cd %s;zip -j %s %s/*'%(dump_path, zip_path, txt_path);
              print cmd
              print util.cmd.cmd(cmd);
              print "zip file created: ", util.io.join_path(dump_path, zip_path)
              
            image_path = util.io.join_path(dump_path, ckpt_name, FLAGS.dataset_split_name, "vis", "%s_%s.jpg")
            for i in xrange(data_provider.num_images):
              print 'image %d/%d'%(i + 1, data_provider.num_images)
              image_data, bbox_data, label_data, name = data_provider.get_data();
              
              sdict, bdict = sess.run([rscores, rbboxes], feed_dict = {image:image_data})
              bbox_score = sdict[1][0, :]
              bbox_pred = bdict[1][0, ...]
              write_result(name, image_data, bbox_pred, bbox_score, path = txt_path)
              img_gt = draw_bbox(image_data, bbox_data, label_data, color = util.img.COLOR_GREEN)
              img_pred = draw_bbox(image_data, bbox_pred, bbox_score, util.img.COLOR_RGB_RED)
              util.img.imwrite(image_path%(name, 'pred'), img_pred, rgb = True)
              util.img.imwrite(image_path%(name, 'gt'), img_gt, rgb = True)
            
            create_zip()
            import datasets.deteval
            result_path = util.io.join_path(dump_path, ckpt_name, FLAGS.dataset_split_name, 'fixed_eval.txt')
            datasets.deteval.eval(det_txt_dir = txt_path, gt_txt_dir = data_provider.gt_path, xml_path = xml_path, write_path = result_path);
            
          
if __name__ == '__main__':
    tf.app.run()
