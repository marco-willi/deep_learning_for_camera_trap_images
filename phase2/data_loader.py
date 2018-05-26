from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import xrange
import numpy as np
import tensorflow as tf
import scipy.misc
import matplotlib.image as mpimg

# convert input data to appropriate labels
def _process_label(label):
    info=np.zeros(8,dtype='int32')
    info[0]=int(label[0]);
    #if label[1]=='51+':
    #    info[1]=11
    #elif label[1]=='11-50':
    #    info[1]=10
    #else:
    #    info[1]=int(label[1])-1
    intc= int(label[1])
    if intc<=10:
      info[1]= intc-1
    elif intc>=11 and intc<=50:
      info[1]= 10
    else:
      info[1]= 11
    for i in range(2, 8):
        info[i]=int(label[i])
    return info


# Parse the input file name
def _read_label_file(file, delimiter):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    tokens = line.split(delimiter)
    filepaths.append(tokens[0])
    labels.append(_process_label(tokens[2:]))
  return filepaths, np.vstack(labels)


def img_resize(add):
  try:
    img=mpimg.imread(add)
    img=scipy.misc.imresize(img[0:-100,:],(256,256))
    return img
  except:
    print("Severe Exception for: "+add)
    return np.zeros((256,256,3),dtype=np.uint8)
    #raise

def read_inputs(is_training, args):

  filepaths, labels = _read_label_file(args.data_info, args.delimiter)

  filenames = [os.path.join(args.path_prefix,i) for i in filepaths]

  # Create a queue that produces the filenames to read.
  if is_training:
    filename_queue = tf.train.slice_input_producer([filenames, labels], shuffle= args.shuffle, capacity= 1024)
  else:
    filename_queue = tf.train.slice_input_producer([filenames, labels], shuffle= False,  capacity= 1024, num_epochs =1)

  # Read examples from files in the filename queue.
  file_content = tf.read_file(filename_queue[0])
  # Read JPEG or PNG or GIF image from file
  reshaped_image = tf.to_float(tf.image.decode_jpeg(file_content, channels=args.num_channels))
  # Resize image to 256*256
  #reshaped_image = tf.image.resize_images(reshaped_image, args.load_size)
  reshaped_image = tf.cast(tf.py_func(img_resize, [filename_queue[0]], tf.uint8),tf.float32)

  label = tf.cast(filename_queue[1], tf.int64)
  img_info = filename_queue[0]

  if is_training:
    reshaped_image = _train_preprocess(reshaped_image, args)
  else:
    reshaped_image = _test_preprocess(reshaped_image, args)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(5000*
                           min_fraction_of_examples_in_queue)

  print ('Filling queue with %d images before starting to train. '
         'This may take some times.' % min_queue_examples)
  batch_size = args.chunked_batch_size if is_training else args.batch_size

  # Load images and labels with additional info
  if hasattr(args, 'save_predictions') and args.save_predictions is not None:
    images, label_batch, info = tf.train.batch(
        [reshaped_image, label, img_info],
        batch_size= batch_size,
        num_threads=args.num_threads,
        capacity=min_queue_examples+3 * batch_size,
        allow_smaller_final_batch=True if not is_training else False)
    return images, label_batch, info
  else:
    images, label_batch = tf.train.batch(
        [reshaped_image, label],
        batch_size= batch_size,
        allow_smaller_final_batch= True if not is_training else False,
        num_threads=args.num_threads,
        capacity=min_queue_examples+3 * batch_size)
    return images, label_batch


def _train_preprocess(reshaped_image, args):
  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  reshaped_image = tf.random_crop(reshaped_image, [args.crop_size[0], args.crop_size[1], args.num_channels])

  # Randomly flip the image horizontally.
  reshaped_image = tf.image.random_flip_left_right(reshaped_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  reshaped_image = tf.image.random_brightness(reshaped_image,
                                               max_delta=63)
  # Randomly changing contrast of the image
  reshaped_image = tf.image.random_contrast(reshaped_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  reshaped_image = tf.image.per_image_standardization(reshaped_image)

  # Set the shapes of tensors.
  reshaped_image.set_shape([args.crop_size[0], args.crop_size[1], args.num_channels])

  return reshaped_image


def _test_preprocess(reshaped_image, args):

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         args.crop_size[0], args.crop_size[1])

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([args.crop_size[0], args.crop_size[1], args.num_channels])

  return float_image
