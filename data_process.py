# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings('ignore')

import os
import h5py
import timeit
import argparse
import numpy as np

import PIL.Image
from tqdm import tqdm

import tensorflow as tf



class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert(os.path.isdir(self.tfrecord_dir))
        
    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): 
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.NONE)
            
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.compat.v1.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))
            
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()



def create_from_h5py(tfrecord_dir, h5py_file, mode, shuffle = 0):
    print('Loading h5py file from "%s"' % h5py_file)
    with h5py.File(h5py_file, "r") as hf:
        images = hf[mode][:]  
    assert images[0].shape == (3, 64, 64)
    num_img = images.shape[0]
    
    with TFRecordExporter(tfrecord_dir, num_img) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(num_img)
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
            
            
def ffhq(h5py_file, img_dir, resolution=64, nc = 3):
    
    image_filenames = []
    for root,dirs,files in os.walk(img_dir):
        for file in files:
            image_filenames.append(os.path.join(root, file))
    img_files = sorted(image_filenames)
    # delete .txt file
    if '.txt' in img_files[-1]:
        _ = img_files.pop(-1)

    num_img = len(img_files)
    print('total images>>>', num_img)
    
    
    images = np.zeros((num_img, nc, resolution, resolution), dtype=np.uint8)
    for idx, img_file in tqdm(enumerate(img_files)):
        img = np.asarray(PIL.Image.open(img_file))
        assert img.shape == (128, 128, 3)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
        img = np.asarray(img)
        img = img.transpose(2, 0, 1) 
        images[idx] = img
    
    
    with h5py.File(h5py_file, "w") as f:
        f.create_dataset('images', data = images, dtype='uint8')

def make_sub_dataset(h5py_file, selected_imgs_file, shuffled_idx_file, num_images, random_seed):
    with h5py.File(h5py_file, "r") as f:
        images = f['images'][:]  
        print(images.shape)


    np.random.seed(random_seed)       
    shuffled_idx = np.arange(images.shape[0])
    print(shuffled_idx.shape)
    np.random.shuffle(shuffled_idx)    
    np.save(shuffled_idx_file,shuffled_idx) 
    print(shuffled_idx[:10])


    images_selected = images[shuffled_idx[:num_images]]

    with h5py.File(selected_imgs_file, "w") as f:
        f.create_dataset('images', data = images_selected, dtype='uint8')
    
def main(args):
    
    start = timeit.default_timer()
    # Set seed
    random_seed=args.seed
    np.random.seed(random_seed)
    if not os.path.exists(args.base_dir):
        os.mkdir(args.base_dir)
    # transform dataset into .h5py dataset    
    print('transform dataset into .h5py dataset...')
    h5py_file=args.base_dir + 'ffhq_all.h5py'
    ffhq(h5py_file, args.img_dir, args.resolution, nc = 3)
  
   # Transforming the dataset into a .h5py dataset, the default of the shuffled_idx_file is sequential.
    if args.sub_dataset != 'True' or args.tf_dataset !='True':
        shuffled_idx_file=args.base_dir + f'ffhq_{args.num_images}_idx.npy'
        shuffled_idx = np.arange(args.num_images)
        print(shuffled_idx.shape)
        np.save(shuffled_idx_file,shuffled_idx) 

        
    # Spliting a dataset into  samples and nonmember samples (random select)
    
    if args.sub_dataset == 'True':
        print('make sub dataset...')
        selected_imgs_file=args.base_dir + f'ffhq_{args.num_images}.h5py'
        shuffled_idx_file=args.base_dir + f'ffhq_{args.num_images}_idx.npy'
        make_sub_dataset(h5py_file, selected_imgs_file, 
                                  shuffled_idx_file, args.num_images, 
                                  random_seed)
    
    # Making tfrecords dataset for training target models
    
    if args.sub_dataset == 'True' and args.tf_dataset=='True':
        print('make tfrecords dataset...')
        tfrecord_dir = args.base_dir + f'tfrecord_dir_ffhq_{args.num_images}'
        if not os.path.exists(tfrecord_dir):
            os.mkdir(tfrecord_dir)
    
        create_from_h5py(tfrecord_dir, selected_imgs_file, mode = 'images')        

    stop = timeit.default_timer()
    print("Time elapses: {}s".format(stop - start))    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Preparation')
    
   
    parser.add_argument('--seed', default=2022, type=int, help='')
    parser.add_argument('--resolution', default=64, type=int, help='')
    parser.add_argument('--num_images', default=1000, type=int, help='The number of member(training) samples.')
    

    parser.add_argument('--img_dir', default='./', type=str, help='Path for the raw image folder.')   
    parser.add_argument('--base_dir', default='./', type=str, help='Path for the base folder to save results.') 

    parser.add_argument('--sub_dataset', default='True', type=str, help='Spliting a dataset into member(training) samples and nonmember samples.')
    parser.add_argument('--tf_dataset', default='True', type=str, help='Making tfrecords dataset for training.')
       
    args, other_args = parser.parse_known_args()
    
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    ### RUN
    main(args)
    