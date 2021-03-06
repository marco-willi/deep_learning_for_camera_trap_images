"""Evaluating a trained model on the test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import json

import numpy as np
import tensorflow as tf
import argparse
import arch
import data_loader
import sys


def predict(args):

  # Building the graph
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Get images and labels.
    images, urls = data_loader.read_inputs(False, args, False)
    # Performing computations on a GPU
    with tf.device('/gpu:0'):
      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = arch.get_model(images, 0.0, False, args)

      # Information about the predictions for saving in a file

      # Species Identification
      top5_id = tf.nn.top_k(tf.nn.softmax(logits[0]), 5)
      top5ind_id= top5_id.indices
      top5val_id= top5_id.values
      # Count
      top3_cn = tf.nn.top_k(tf.nn.softmax(logits[1]), 3)
      top3ind_cn= top3_cn.indices
      top3val_cn= top3_cn.values
      # Additional Attributes (e.g. description)
      top1_bh = [None]*6
      top1ind_bh = [None]*6
      top1val_bh = [None]*6

      for i in xrange(0,6):
        top1_bh[i]= tf.nn.top_k(tf.nn.softmax(logits[i+2]), 1)
        top1ind_bh[i]= top1_bh[i].indices
        top1val_bh[i]= top1_bh[i].values

      # For reading the snapshot files from file
      saver = tf.train.Saver(tf.global_variables())

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.summary.merge_all()

      summary_writer = tf.summary.FileWriter(args.log_dir, g)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      ckpt = tf.train.get_checkpoint_state(args.log_dir)

      # Load the latest model
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)

      else:
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      step = 0

      # Output file to save predictions and their confidences
      out_file = open(args.save_predictions,'w')
      out_file.write('{')
      first_row = True
      while step < args.num_batches and not coord.should_stop():

        urls_values, top5guesses_id, top5conf, top3guesses_cn, top3conf, top1guesses_bh, top1conf= sess.run([urls, top5ind_id, top5val_id, top3ind_cn, top3val_cn, top1ind_bh, top1val_bh])
        for i in xrange(0,urls_values.shape[0]):
          step_result = {'path': urls_values[i],
                         'top_n_pred':  [int(np.asscalar(item)) for item in top5guesses_id[i]],
                         'top_n_conf': [round(float(np.asscalar(item)), 4) for item in top5conf[i]],
                         'top_n_pred_count':  [int(np.asscalar(item)) for item in top3guesses_cn[i]],
                         'top_n_conf_count': [round(float(np.asscalar(item)), 4) for item in top3conf[i]],
                         'top_pred_standing': int(np.asscalar(top1guesses_bh[0][i])),
                         'top_pred_resting': int(np.asscalar(top1guesses_bh[1][i])),
                         'top_pred_moving': int(np.asscalar(top1guesses_bh[2][i])),
                         'top_pred_eating': int(np.asscalar(top1guesses_bh[3][i])),
                         'top_pred_interacting': int(np.asscalar(top1guesses_bh[4][i])),
                         'top_pred_young_present': int(np.asscalar(top1guesses_bh[5][i])),
                         'top_conf_standing': round(float(np.asscalar(top1conf[0][i])), 4),
                         'top_conf_resting': round(float(np.asscalar(top1conf[1][i])), 4),
                         'top_conf_moving': round(float(np.asscalar(top1conf[2][i])), 4),
                         'top_conf_eating': round(float(np.asscalar(top1conf[3][i])), 4),
                         'top_conf_interacting': round(float(np.asscalar(top1conf[4][i])), 4),
                         'top_conf_young_present': round(float(np.asscalar(top1conf[5][i])), 4)
                         }
          if first_row:
            out_file.write('"' + str(int(step*args.batch_size+i+1)) + '":')
            first_row = False
          else:
            out_file.write(',\n"' + str(int(step*args.batch_size+i+1)) + '":')
          json.dump(step_result, out_file)

          out_file.flush()
        print("Finished predicting batch %s / %s" % (step, args.num_batches))
        sys.stdout.flush()

        step += 1
      out_file.write('}')
      out_file.close()

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      coord.request_stop()
      coord.join(threads)

def main():
  parser = argparse.ArgumentParser(description='Process Command-line Arguments')
  parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk')
  parser.add_argument('--crop_size', nargs= 2, default= [224,224], type= int, action= 'store', help= 'The width and height of images after random cropping')
  parser.add_argument('--batch_size', default= 512, type= int, action= 'store', help= 'The testing batch size')
  parser.add_argument('--num_classes', default= [48, 12, 2, 2, 2, 2, 2, 2] , type=int, nargs= '+', help= 'The number of classes')
  parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images')
  parser.add_argument('--num_batches' , default=-1 , type= int, action= 'store', help= 'The number of batches of data')
  parser.add_argument('--path_prefix' , default='/project/EvolvingAI/mnorouzz/Serengiti/resized', action= 'store', help= 'The prefix address for images')
  parser.add_argument('--delimiter' , default=',', action = 'store', help= 'Delimiter for the input files')
  parser.add_argument('--data_info'   , default= 'gold_expert_info.csv', action= 'store', help= 'File containing the addresses and labels of testing images')
  parser.add_argument('--num_threads', default= 20, type= int, action= 'store', help= 'The number of threads for loading data')
  parser.add_argument('--architecture', default= 'resnet', help='The DNN architecture')
  parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
  parser.add_argument('--log_dir', default= None, action= 'store', help='Path for saving Tensorboard info and checkpoints')
  parser.add_argument('--save_predictions', default= None, action= 'store', help= 'Save predictions of the networks along with their confidence in the specified file')

  args = parser.parse_args()
  args.num_samples = sum(1 for line in open(args.data_info))
  if args.num_batches==-1:
    if(args.num_samples%args.batch_size==0):
      args.num_batches= int(args.num_samples/args.batch_size)
    else:
      args.num_batches= int(args.num_samples/args.batch_size)+1

  print(args)

  predict(args)


if __name__ == '__main__':
  main()
