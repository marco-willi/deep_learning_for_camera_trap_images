"""Predict on new data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import argparse
import arch
import data_loader
import sys
import json


def predict(args):

  # Building the graph
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Get images
    images, urls = data_loader.read_inputs(False, args, has_labels=False)
    # Performing computations on a GPU
    with tf.device('/gpu:0'):
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = arch.get_model(images, 0.0, False, args)

        # Calculate predictions
        topn = tf.nn.top_k(tf.nn.softmax(logits), args.top_n)
        topnind = topn.indices
        topnval = topn.values

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
        print('Checkpoint not found: '+args.log_dir)
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()

      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      step = 0

      out_file = open(args.save_predictions,'w')
      out_file.write('{')
      first_row = True
      while step < args.num_batches and not coord.should_stop():
        urls_values, topnguesses, topnconf = sess.run([urls, topnind, topnval])
        for i in xrange(0,urls_values.shape[0]):
            step_result = {
                'path': urls_values[i],
                'top_n_pred':  [int(item) for item in topnguesses[i]],
                'top_n_conf': [round(float(item), 4) for item in topnconf[i]]
                           }
            if first_row:
              out_file.write('"' + str(int(step*args.batch_size+i+1)) + '":')
              first_row = False
            else:
              out_file.write(',"' + str(int(step*args.batch_size+i+1)) + '":')
            json.dump(step_result, out_file)
            out_file.write('\n')
            out_file.flush()
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
  parser.add_argument('--num_classes', default= 2, type= int, action= 'store', help= 'The number of classes')
  parser.add_argument('--top_n', default= 2, type= int, action= 'store', help= 'Top N accuracy')
  parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images')
  parser.add_argument('--num_batches' , default=-1 , type= int, action= 'store', help= 'The number of batches of data')
  parser.add_argument('--path_prefix' , default='/project/EvolvingAI/mnorouzz/Serengiti/EmptyVsFullEQ/', action= 'store', help= 'The prefix address for images')
  parser.add_argument('--delimiter' , default=',', action = 'store', help= 'Delimiter for the input files')
  parser.add_argument('--data_info'   , default= 'EF_val.csv', action= 'store', help= 'File containing the addresses prediction images')
  parser.add_argument('--num_threads', default= 20, type= int, action= 'store', help= 'The number of threads for loading data')
  parser.add_argument('--architecture', default= 'resnet', help='The DNN architecture')
  parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
  parser.add_argument('--log_dir', default= None, action= 'store', help='Path for saving Tensorboard info and checkpoints')
  parser.add_argument('--save_predictions', default= None, action= 'store', help= 'Save top-5 predictions of the networks along with their confidence in the specified file')

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
