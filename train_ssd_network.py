# Copyright 2016 Paul Balanca. All Rights Reserved.
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
"""Generic training script that trains a SSD model using a given dataset."""
import numpy
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory

import tf_utils
import util
slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_boolean(
    'loss_weighted_blocks', False, 'Use different weights for different blocks when calculating loss')
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.25, 'Matching threshold in the loss function.')

# =========================================================================== #
# General Flags.
# =========================================================================== #

tf.app.flags.DEFINE_string('train_dir', None,'Directory where checkpoints and event logs are written to.')
#tf.app.flags.DEFINE_integer('num_clones', 1,'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')
                        
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_float(
    'min_object_covered', 0.5, 'The cropped area of the image must contain at least this fraction of any bounding box supplied, used in preprocessing')
tf.app.flags.DEFINE_float(
    'min_width_covered', 0.5, 'The minimal fraction of covered width needed to keep that bbox in the cropped image , used in preprocessing')
tf.app.flags.DEFINE_float(
    'min_height_covered', 0.95, 'The minimal fraction of covered height needed to keep that bbox in the cropped image, used in preprocessing')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 5,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 300,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', -1, 'GPU memory fraction to use. If -1, allow_growth is used.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'fixed',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', None, 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', None, 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')


FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
#    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(log_file = 'train_on_%s.log'%(FLAGS.dataset_name), log_path = FLAGS.train_dir, mode='a')
    
    # use all available gpus
    gpus = util.tf.get_available_gpus();
    num_clones = len(gpus);
    if num_clones > 1 and FLAGS.batch_size % num_clones != 0:
        raise ValueError('If multi gpus are used, the batch_size should be a multiple of the number of gpus.')
    batch_size = FLAGS.batch_size / num_clones;
    print "%d images per GPU"%(batch_size)
    
    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones = num_clones,
            clone_on_cpu = FLAGS.clone_on_cpu,
            replica_id=0,#no use if on a single machine
            num_replicas = 1,#no use if on a single machine
            num_ps_tasks= 0)#no use if on a single machine
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()
        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)
        util.proc.set_proc_name(FLAGS.model_name + '_' + FLAGS.dataset_name)
        # Select the preprocessing function.
        
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        from preprocessing import preprocessing_factory
        from preprocessing import ssd_vgg_preprocessing
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        tf_utils.print_configuration(FLAGS.__flags, ssd_params,
                                     dataset.data_sources, FLAGS.train_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        
        
        with tf.device(deploy_config.inputs_device()):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
            image = tf.identity(image, 'input_image')
            # Pre-processing image, labels and bboxes.
            image, glabels, gbboxes, _ = image_preprocessing_fn(image, glabels, gbboxes,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT)
            image = tf.identity(image, 'processed_image')
            # Encode groundtruth labels and bboxes.
            gclasses, glocalizations, gscores = ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors, match_threshold = FLAGS.match_threshold)
            batch_shape = [1] + [len(ssd_anchors)] * 3
            # Training batches and queue.
#                tf_utils.reshape_list([image, gclasses, glocalizations, gscores]),
            b_image, b_gclasses, b_glocalizations, b_gscores = tf.train.batch(
                [image, gclasses, glocalizations, gscores],
                batch_size = batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_image, b_gclasses, b_glocalizations, b_gscores],
                capacity=2 * num_clones)

        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        t_batch_size = batch_size;
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple
            clones of network_fn."""
            # Dequeue batch.
            #b_image, b_gclasses, b_glocalizations, b_gscores = tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)
            b_image, b_gclasses, b_glocalizations, b_gscores = batch_queue.dequeue()
            t_batch_size = tf.shape(b_image)[0] * num_clones
            # Construct SSD network.
            arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                          data_format=DATA_FORMAT)
            with slim.arg_scope(arg_scope):
                confidences, localizations, logits, end_points = ssd_net.net(b_image, is_training=True) 
            # Add loss function.
            #confidences = tf.Print(confidences, ["shape of confidences(b, N, n_cls)", tf.shape(confidences)])
            #b_gscores = tf.Print(b_gscores, ["shape of b_gscores:(b, N, 1)", tf.shape(b_gscores)])
            #localizations = tf.Print(localizations, ["shape of localizations(b, N, 4)", tf.shape(localizations)])
            #b_glocalizations = tf.Print(b_glocalizations, ["shape of b_glocalizations(b, N, 4)", tf.shape(b_glocalizations)])
            ssd_net.losses(confidences, logits, localizations,
                           b_gclasses, b_glocalizations, b_gscores,
                           match_threshold=FLAGS.match_threshold,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha)
            return end_points

        tf.summary.scalar('batch_size_per_gpu', t_batch_size)
        tf.summary.scalar('batch_size', t_batch_size * num_clones)
        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # =================================================================== #
        # Add summaries from first clone.
        # =================================================================== #
        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))
        # Add summaries for losses and extra losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        for loss in tf.get_collection('EXTRA_LOSSES', first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        # =================================================================== #
        # Configure the moving averages.
        # =================================================================== #
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None
        
        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                             dataset.num_samples,
                                                             global_step)
            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = tf_utils.get_variables_to_train(FLAGS)

        # and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        summaries |= set(model_deploy._add_gradients_summaries(clones_gradients))
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))
        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        
        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        if FLAGS.gpu_memory_fraction < 0:
            config.gpu_options.allow_growth = True
        elif FLAGS.gpu_memory_fraction > 0:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
        print config
        saver = tf.train.Saver(max_to_keep=500,
                               write_version=2,
                               pad_step_number=False)

        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            master='',
            is_chief=True,
            init_fn=tf_utils.get_init_fn(FLAGS),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs= FLAGS.save_summaries_secs,
            saver=saver,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=config,
            sync_optimizer=None
            )


if __name__ == '__main__':
    tf.app.run()
