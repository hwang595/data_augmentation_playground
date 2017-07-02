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
"""A binary to train Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_preparation.cifar10_input as cifar10
import resnet_train

FLAGS = tf.app.flags.FLAGS

def main(_):
  cifar10.maybe_download_and_extract()
  train_set, train_labels = cifar10.prepare_train_data(padding_size=FLAGS.padding_size)
  # do some debugging tests here
  #print(train_set.shape, train_labels.shape)
  #print("==========================================")
  #exit()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  resnet_train.train(train_set, train_labels)

if __name__ == '__main__':
  tf.app.run()
