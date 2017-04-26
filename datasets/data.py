#encoding=utf-8
import util;
import numpy as np;
class ICDAR2013TrainingData(object):
    def __init__(self):
        root_dir = '~/dataset_nfs/ICDAR2015/Challenge2.Task123/'
        training_data_dir = 'Challenge2_Training_Task12_Images'
        training_gt_dir = 'Challenge2_Training_Task1_GT'
#        test_data_dir = 'Challenge2_Test_Task12_Images'
#        test_gt_dir = 'Challenge2_Test_Task1_GT'
        self.image_idx = -1;
        self.training_images, self.training_bboxes = self.get_images_and_gt(util.io.join_path(root_dir, training_data_dir), util.io.join_path(root_dir, training_gt_dir));
#        self.test_images, self.test_bboxes = self.get_images_and_gt(util.io.join_path(root_dir, test_data_dir), util.io.join_path(root_dir, test_gt_dir));
        images = self.training_images# + self.test_images;
        bboxes = self.training_bboxes# + self.test_bboxes;
        self.data = [[img, bboxes] for(img, bboxes) in zip(images, bboxes)]

    def get_images_and_gt(self, data_path, gt_path):
        image_names = util.io.ls(data_path, '.jpg');
        print "%d images found in %s"%(len(image_names), data_path);
        images = [];
        bboxes = [];
        for idx, image_name in enumerate(image_names):
            path = util.io.join_path(data_path, image_name);
            print "\treading image: %d/%d"%(idx, len(image_names));
            image = util.img.imread(path, rgb = True);
            images.append(image);
            image_name = util.str.split(image_name, '.')[0];
            gt_name = 'gt_' + image_name + '.txt';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            lines = util.io.read_lines(gt_filepath);
            bbox_gt = []
            h, w = image.shape[0:2];
            h *= 1.0;
            w *= 1.0;
            for line in lines:
                gt = util.str.remove_all(line, ',');
                gt = util.str.split(gt, ' ');
                box =[int(gt[i]) for i in range(4)];
                x1, y1, x2, y2  = box;
                box = [y1 / h, x1 / w, y2 / h,  x2 / w];
                bbox_gt.append(box);
            bboxes.append(bbox_gt);
#            if idx >= 1:
#                break;
            
        return images, bboxes;
        
    def vis_data(self):
        for image, bboxes in self.data:
            for bbox in bboxes:
                x1, y1, x2, y2  = bbox;
                util.img.rectangle(image, (x1, y1), (x2, y2), color = util.img.COLOR_WHITE, border_width = -1);
            
            util.img.imshow("", image);
    
    def get_data(self):
        self.image_idx += 1;
        if self.image_idx >= len(self.data):
            util.np.shuffle(self.data);
            self.image_idx = 0;
        image, bboxes = self.data[self.image_idx];
        labels = [1] * len(bboxes);
        labels = np.reshape(labels, [-1, 1]);
        return image, [bboxes], labels;

if __name__ == "__main__":


    import numpy as np
    import tensorflow as tf
    tf.device('/cpu:0')
    import time
    import util
    util.mod.add_to_path('..')
    import pdb
    pdb.set_trace()
    from preprocessing.preprocessing_factory  import get_preprocessing
    data_provider = ICDAR2013TrainingData();
    
    util.proc.set_proc_name('proc-test');
    
    fn = get_preprocessing(True);
    with tf.Graph().as_default():
        sess = tf.Session();
        sess.as_default();
        out_shape = [150, 150]
        images = tf.placeholder("float", name = 'images', shape = [None, None, 3])
        bboxes = tf.placeholder("float", name = 'bboxes', shape = [1, None, 4])
        labels = tf.placeholder('int32', name = 'labels', shape = [None, 1])
        
        sampled_image, sampled_labels, sampled_bboxes = fn(images, labels, bboxes, out_shape);
        step = 0;
        data = []
        while step < 10:
            step += 1;
            start = time.time();
            image_data, bbox_data, label_data = data_provider.get_data();
            feed_dict = {images: image_data, labels: label_data, bboxes: bbox_data}
            I, L, B = sess.run([sampled_image, sampled_labels, sampled_bboxes], feed_dict = feed_dict)
            I = np.asarray(I, dtype = np.uint8);
            util.cit(I)
            B *= 150;
            I_box = I.copy()
            for bbox in B:
                y1, x1, y2, x2 = bbox
                util.img.rectangle(I_box, (x1, y1), (x2, y2), color = util.img.COLOR_WHITE)
            end = time.time();
            util.plt.show_images(images = [I, I_box], save = True, show = False, path = '~/temp_nfs/no-use/%d.jpg'%(step))
#            data.append([image_data, I_, L]);
#            util.io.dump('test.pkl', data);
