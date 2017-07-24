# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple GPU's with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time
import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import metrics_impl
import resnet
import data_preparation.cifar10_input as cifar10

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
MOMENTUM = 0.9

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 80000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")


# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 60000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_float('init_lr', 0.1, '''Initial learning rate''')

# this hyper parameter is only used for resnet
tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def train(training_set, training_labels, sampled_train_images, sampled_train_labels):
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    # get num of examples in training set
#    dataset_num_examples = training_set.shape[0]

    # Calculate the learning rate schedule.
#    num_batches_per_epoch = (dataset_num_examples / FLAGS.batch_size)

#    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    '''
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    '''
    lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    # Create an optimizer that performs gradient descent.
    #opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr_placeholder, MOMENTUM)

    #fetch the data batch from training set
    images, labels = cifar10.placeholder_inputs(FLAGS.batch_size)
    logits = resnet.inference(images, FLAGS.num_residual_blocks, reuse=False)

    #calc the loss and gradients
    loss = resnet.loss(logits, labels)
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + regu_losses)

    grads = opt.compute_gradients(total_loss)

    # Apply the gradients to adjust the shared variables.
    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradients_op]):
      train_op = tf.identity(total_loss, name='train_op')

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # For testing trained model
#    test_size = testset.num_examples
#    test_images_placeholder, test_labels_placeholder = mnist.placeholder_inputs(FLAGS.batch_size)
#    logits_test = mnist.inference(test_images_placeholder, train=False)
    #pred = mnist.predictions(logits_test)
    validation_accuracy = tf.reduce_sum(resnet.evaluation(logits, labels)) / tf.constant(FLAGS.batch_size)
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # these two parameters is used to measure when to enter next epoch
    local_data_batch_idx = 0
    epoch_counter = 0

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    for step in range(FLAGS.max_steps):
      # change the API for new aug method
      training_set, training_labels, epoch_counter, local_data_batch_idx, feed_dict = cifar10.fill_feed_dict(
                training_set, 
                training_labels, 
                images, 
                labels, 
                FLAGS.batch_size, 
                local_data_batch_idx, 
                epoch_counter,
                FLAGS.init_lr, lr_placeholder,
                sampled_train_images, sampled_train_labels)

      start_time = time.time()
      _, loss_value, acc = sess.run([train_op, total_loss, validation_accuracy], feed_dict=feed_dict)

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f '
                    'sec/batch); acc=%.4f')
      tf.logging.info(format_str % (datetime.now(), step, loss_value,
                          examples_per_sec, duration, acc))
      tf.logging.info("Data batch index: %s, Current epoch idex: %s" % (str(epoch_counter), str(local_data_batch_idx)))
      
      if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
        FLAGS.init_lr = 0.1 * FLAGS.init_lr

      if step % 2000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)