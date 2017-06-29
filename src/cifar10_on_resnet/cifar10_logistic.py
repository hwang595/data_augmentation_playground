import tensorflow as tf
import data_preparation.cifar10_input as cifar10
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.linear import LinearClassifier
from sklearn.linear_model import LogisticRegression
import copy
import random

def load_cifar10():
	cifar10.maybe_download_and_extract()
	train_set, train_labels = cifar10.prepare_train_data(0)
	test_set, test_labels = cifar10.read_validation_data()
#	train_set = train_data.reshape((train_data.shape[0], train_data.shape[1]*train_data.shape[2]*train_data.shape[3]))
#	test_set = test_data.reshape((test_data.shape[0], test_data.shape[1]*test_data.shape[2]*test_data.shape[3]))
	return train_set, train_labels, test_set, test_labels

def construct_feature_col(col_num=None):
	feature_col = []
	feature_dummy = []
	feature_name_base = "pixel_"
	for col_idx in range(col_num):
		feature_col.append(tf.contrib.layers.real_valued_column(feature_name_base+str(col_idx)+"th"))
		feature_dummy.append(feature_name_base+str(col_idx)+"th")
	return feature_col, feature_dummy

def extract_for_binary(train_set=None, train_labels=None, test_set=None, test_labels=None):
	binary_indices_train = []
	binary_indices_test = []
	for i in range(len(train_set)):
		if train_labels[i] == 0 or train_labels[i] == 1:
			binary_indices_train.append(i)
	for i in range(len(test_set)):
		if test_labels[i] == 0 or test_labels[i] == 1:
			binary_indices_test.append(i)
	train_set_binary = np.take(train_set, binary_indices_train, axis=0)
	train_label_binary = np.take(train_labels, binary_indices_train)
	test_set_binary = np.take(test_set, binary_indices_test, axis=0)
	test_label_binary = np.take(test_labels, binary_indices_test)
	return train_set_binary, train_label_binary, test_set_binary, test_label_binary

def input_fn(data_set=None, labels=None, feature_col=None):
	# Creates a dictionary mapping from each continuous feature column name (k) to
	# the values of that column stored in a constant Tensor.
	feature_columns = {k_val: tf.constant(data_set[:, k_idx]) for k_idx, k_val in enumerate(feature_col)}
	label = tf.constant(labels)
	return feature_columns, label

def down_sample(data_set=None, labels=None, down_sample_num=None):
	down_sample_indices = np.random.randint(low=0, high=data_set.shape[0], size=down_sample_num)
	down_samples = np.take(data_set, down_sample_indices, axis=0)
	down_sample_labels = np.take(labels, down_sample_indices)
	return down_samples, down_sample_labels

def flip_up_down(batch=None):
	batch=batch.reshape((batch.shape[0], int(np.sqrt(batch.shape[1])), int(np.sqrt(batch.shape[1]))))
	for i in range(len(batch)):
		batch[i] = np.fliplr(batch[i])
	return batch

def aug_data_set(ori_data, ori_labels, times_expand=1):
  aug_data_list = []
  new_data=ori_data
  new_label=ori_labels
  for time_aug in range(times_expand):
    crop_data = random_crop(ori_data, crop_shape=(32, 32), padding=10)
    aug_data_list.append(crop_data)
    new_data = np.concatenate((new_data,aug_data_list[time_aug]),axis=0)
    new_label = np.concatenate((new_label,ori_labels), axis=0)
  return new_data, new_label

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
  return np.array(new_batch)

if __name__ == "__main__":
	times_expand_list = [i for i in range(1, 15)]
	train_set, train_labels, test_set, test_labels = load_cifar10()
	train_set_binary, train_labels_binary, test_set_binary, test_labels_binary = extract_for_binary(train_set, train_labels, test_set, test_labels)
	test_set_binary = test_set_binary.reshape(test_set_binary.shape[0], test_set_binary.shape[1]*test_set_binary.shape[2]*test_set_binary.shape[3])
	accuracy_list = []
	
	sample_data, sample_labels=down_sample(train_set_binary, train_labels_binary, down_sample_num=1000)
	for item in times_expand_list:
		train_set_new, train_labels_new = aug_data_set(sample_data, sample_labels, times_expand=item)
#	sample_data = sample_data.reshape(sample_data.shape[0], sample_data.shape[1]*sample_data.shape[2]*sample_data.shape[3])
#	train_set_binary = train_set_binary.reshape(train_set_binary.shape[0], train_set_binary.shape[1]*train_set_binary.shape[2]*train_set_binary.shape[3])
		train_set_new = train_set_new.reshape(train_set_new.shape[0], train_set_new.shape[1]*train_set_new.shape[2]*train_set_new.shape[3])
		print(train_set_new.shape)
		perceptron_classifier = LogisticRegression(max_iter=100)
		perceptron_classifier.fit(train_set_new, train_labels_new)
		accuracy = perceptron_classifier.score(test_set_binary, test_labels_binary)
		#print(accuracy)
		accuracy_list.append(accuracy)
	print("Accuracy: ")
	print(accuracy_list)

#	train_set_new, train_labels_new = aug_data_set(train_set, train_labels, times_expand=1)

	# extract images whose label is 0 or 1 to do binary classification
	#train_set_binary, train_label_binary, test_set_binary, test_label_binary = extract_for_binary(train_set, train_labels, test_set, test_labels)
	#sample_data, sample_labels=down_sample(train_set_binary, train_label_binary, down_sample_num=1000)
	
	# test data augmentation
	'''
	acc_list = []
	for i in times_expand_list:
		new_data, new_labels = aug_data_set(sample_data, sample_labels, times_expand=i, aug_type="flip_up_down")
		print(new_data.shape)
		perceptron_classifier = LogisticRegression(max_iter=500)
		perceptron_classifier.fit(new_data, new_labels)
		accuracy = perceptron_classifier.score(test_set_binary, test_label_binary)
		acc_list.append(accuracy)
	print(acc_list)
	'''


#	for i in range(len(prediction)):
#		print(prediction[i], test_set[i])
	'''
	feature_col, feature_dummy = construct_feature_col(784)
	# Init an estimator using the default optimizer.
	print(sample_data.shape)
	estimator = LinearClassifier(
    	feature_columns=feature_col)
	estimator.fit(input_fn=lambda: input_fn(sample_data, sample_labels, feature_dummy), steps=10)
	results=estimator.evaluate(input_fn=lambda: input_fn(test_set_binary, test_label_binary, feature_dummy), steps=1)
	for key in sorted(results):
		print("%s: %s" % (key, results[key]))
	'''