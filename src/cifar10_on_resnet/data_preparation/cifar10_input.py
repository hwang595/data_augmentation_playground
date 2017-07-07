# Original Author: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
# Modified by: hwang
# created on: Jun 29, 2017
import tarfile
from six.moves import urllib
import sys
import numpy as np
import cPickle
import os
import tensorflow as tf
import random
#import cv2

data_dir = 'cifar10_data'
full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?

NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH


def maybe_download_and_extract():
    '''
    Will download and extract the cifar10 data automatically
    :return: nothing
    '''
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#        print()
        statinfo = os.stat(filepath)
#        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path, is_random_label):
    '''
    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays

    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''
    fo = open(path, 'rb')
    dicts = cPickle.load(fo)
    fo.close()

    data = dicts['data']
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label


def read_in_all_images(address_list, shuffle=True, is_random_label = False):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print 'Reading images from ' + address
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


    if shuffle is True:
        print 'Shuffling'
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label

def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max(np.std(image_np[i, ...]), int(1.0 / np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)))
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch

def random_crop(batch, crop_shape, padding=None):
  oshape = np.shape(batch[0])
  if padding:
    oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
  new_batch = []
  npad = ((padding, padding), (padding, padding), (0, 0))
  for i in range(len(batch)):
    new_batch.append(batch[i])
    if padding:
      new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                          mode='constant', constant_values=0)
    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                              nw:nw + crop_shape[1]]
    r = np.linalg.norm(np.subtract(batch[i], new_batch[i]))
    new_batch[i]=random_noise(img, mode='gaussian', var=r)
  return np.array(new_batch)

def add_noise_wrt_distance(batch, crop_shape, padding=None):
  oshape = np.shape(batch[0])
  if padding:
    oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
  new_batch = []
  npad = ((padding, padding), (padding, padding), (0, 0))
  var_list = []
  for i in range(len(batch)):
#    batch[i] = normalize(batch[i])
    new_batch.append(batch[i])
    if padding:
      new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                          mode='constant', constant_values=0)
    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                              nw:nw + crop_shape[1]]
    r = np.linalg.norm(np.subtract(batch[i], new_batch[i]))
    var_list.append(r)
  std_var_list = np.array(var_list) / np.linalg.norm(np.array(var_list))
  for i in range(len(std_var_list)):
    std_var_list[i] = std_var_list[i] * 10
  for i in range(len(batch)):
    gaussian_noise = np.random.normal(0, std_var_list[i], (batch[i].shape[0], batch[i].shape[1], batch[i].shape[2]))
    new_batch[i] = new_batch[i] + gaussian_noise
#    gaussian_noise = np.random.normal(0, r, (batch[i].shape[0], batch[i].shape[1], batch[i].shape[2]))
#    new_batch[i] = new_batch[i] + gaussian_noise
  return np.array(new_batch)

def split_subset_wrt_labels(ori_data, ori_labels):
    split_index_table = [[i/10] for i in range(10)]
    split_data_table = []
    for label_idx, label in enumerate(ori_labels):
        split_index_table[int(label)].append(label_idx)
    for idx in range(10):
       data_per_label = np.take(ori_data, split_index_table[idx][1:], axis=0)
       split_data_table.append(data_per_label)
    return split_data_table

def search_data_in_line(data_point=None, other_label_data=None, num_per_label=1, fraction=0.1):
    sample_index = np.random.randint(low=0, high=other_label_data.shape[0], size=num_per_label)
    sampled_data = np.take(other_label_data, sample_index, axis=0)
    new_data_points = []
    for sampled_data_point in sampled_data:
        # calculate epsilon first
        epsilon = fraction * (np.linalg.norm(data_point)/float(np.linalg.norm(sampled_data_point)))
        #print(np.linalg.norm(data_point)-float(np.linalg.norm(sampled_data_point)))
        # generate new data points
        #print(np.linalg.norm(np.multiply(data_point, 1- epsilon)))
        #print(np.linalg.norm(np.multiply(data_point, 1- epsilon)+np.multiply(sampled_data_point, epsilon)))
        new_data_points.append(np.multiply(data_point, 1- epsilon)+np.multiply(sampled_data_point, epsilon))
#        new_data_points.append(data_point)
    return new_data_points

def line_among_labels(ori_data, ori_labels, num_per_label=1, fraction=0.1):
    split_data_table=split_subset_wrt_labels(ori_data, ori_labels)
    new_train_set = []
    new_train_labels = []
    for dp_idx, data_point in enumerate(ori_data):
        ori_data_label = ori_labels[dp_idx]
        for i in range(10):
            if i == ori_data_label:
                continue
            else:
                new_data_points=search_data_in_line(data_point=data_point, other_label_data=split_data_table[i], num_per_label=num_per_label, fraction=fraction)
                for n_d_p in new_data_points:
                    new_train_set.append(n_d_p)
                    new_train_labels.append(ori_labels[dp_idx])
    return np.array(new_train_set), np.array(new_train_labels)

def aug_data_set(ori_data, ori_labels, times_expand=1, aug_type="crop"):
    aug_data_list = []
    new_data=ori_data
    new_label=ori_labels
    for time_aug in range(times_expand):
        if aug_type == 'crop':
            crop_data = add_noise_wrt_distance(ori_data, crop_shape=(32, 32), padding=1)
        elif aug_type == 'line_among_labels':
            crop_data, new_train_labels = line_among_labels(ori_data, ori_labels, num_per_label=1, fraction=0.05)
        elif aug_type == 'fake':
            # this is only used for debug
            crop_data = ori_data
        aug_data_list.append(crop_data)
        new_data = np.concatenate((new_data,aug_data_list[time_aug]),axis=0)
        if aug_type == 'crop':
            new_label = np.concatenate((new_label,ori_labels), axis=0)
        elif aug_type == 'line_among_labels':
            new_label = np.concatenate((new_label,new_train_labels), axis=0)
        elif aug_type == 'fake':
            new_label = np.concatenate((new_label,ori_labels), axis=0)
    return new_data, new_label

def down_sample(data_set=None, labels=None, down_sample_num=None):
    down_sample_indices = np.random.randint(low=0, high=data_set.shape[0], size=down_sample_num)
    down_samples = np.take(data_set, down_sample_indices, axis=0)
    down_sample_labels = np.take(labels, down_sample_indices)
    return down_samples, down_sample_labels

def prepare_train_data(padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(full_data_dir + str(i))
    data, label = read_in_all_images(path_list, is_random_label=TRAIN_RANDOM_LABEL)
    data = whitening_image(data)
#    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
#    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    sampled_train_images, sampled_train_labels = down_sample(data, label, down_sample_num=1024)
    train_set_new, train_labels_new = aug_data_set(sampled_train_images, sampled_train_labels, times_expand=1, aug_type='line_among_labels')    
    print(train_set_new.shape, train_labels_new.shape)
    print("==============================================================")
    order = np.random.permutation(train_set_new.shape[0])
    train_set_new = train_set_new[order, ...]
    train_labels_new = train_labels_new[order]
    return train_set_new, train_labels_new
#    return sampled_train_images, sampled_train_labels

def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir],
                                                       is_random_label=VALI_RANDOM_LABEL)
    validation_array = whitening_image(validation_array)

    return validation_array, validation_labels

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size,))
  return images_placeholder, labels_placeholder

def fill_feed_dict(all_data, all_labels, image_placeholder, label_placeholder, 
                    batch_size, local_data_batch_idx, epoch_counter, lr, lr_placeholder):
    train_batch_data, train_batch_labels, local_data_batch_idx, epoch_counter = generate_augment_train_batch(
                        all_data, all_labels, batch_size, local_data_batch_idx, epoch_counter)
    feed_dict = {image_placeholder: train_batch_data, label_placeholder: train_batch_labels, lr_placeholder: lr}
    return epoch_counter, local_data_batch_idx, feed_dict

def fill_feed_dict_val(val_set, val_labels, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }
    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed = val_set 
    labels_feed = val_labels
    #  print(images_feed.shape, labels_feed.shape)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def generate_augment_train_batch(train_data, train_labels, train_batch_size, local_data_batch_idx, epoch_counter):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    '''
    offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
    batch_data = train_data[offset:offset+train_batch_size, ...]
    batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
    '''
    num_of_instances = train_data.shape[0]
    start = local_data_batch_idx
    local_data_batch_idx += train_batch_size #next time your should start
    if local_data_batch_idx > num_of_instances:
      # Finished epoch
      epoch_counter += 1
      # Shuffle the data
      perm = np.arange(num_of_instances)
      np.random.shuffle(perm)
      train_data = train_data[perm]
      train_labels = train_labels[perm]
      # Start next epoch
      start = 0
      local_data_batch_idx = train_batch_size
      assert train_batch_size <= num_of_instances
    end = local_data_batch_idx
    train_batch = train_data[start:end]
#    train_batch = whitening_image(random_crop_and_flip(train_batch_tmp, padding_size=FLAGS.padding_size))
    batch_labels = train_labels[start:end]
#    tf.logging.info("Batch shapes %s" % str(train_batch.shape))
#    tf.logging.info("Standardized batch shapes %s" % str(whitening_image(train_batch).shape))
    # Most of the time return the non distorted image
    return train_batch, batch_labels, local_data_batch_idx, epoch_counter