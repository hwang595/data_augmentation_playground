from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

IMG_HEIGHT = 8
IMG_WIDTH = 8
IMG_DEPTH = 1

def extract_for_binary(train_set=None, train_labels=None, test_set=None, test_labels=None):
	binary_indices_train = []
	binary_indices_test = []
	for i in range(len(train_set)):
		if train_labels[i] == 8 or train_labels[i] == 9:
			binary_indices_train.append(i)
	for i in range(len(test_set)):
		if test_labels[i] == 8 or test_labels[i] == 9:
			binary_indices_test.append(i)
	train_set_binary = np.take(train_set, binary_indices_train, axis=0)
	train_label_binary = np.take(train_labels, binary_indices_train)
	test_set_binary = np.take(test_set, binary_indices_test, axis=0)
	test_label_binary = np.take(test_labels, binary_indices_test)
	return train_set_binary, train_label_binary, test_set_binary, test_label_binary

def down_sample(data_set=None, labels=None, down_sample_num=None):
	down_sample_indices = np.random.randint(low=0, high=data_set.shape[0], size=down_sample_num)
	down_samples = np.take(data_set, down_sample_indices, axis=0)
	down_sample_labels = np.take(labels, down_sample_indices)
	return down_samples, down_sample_labels

def random_crop(batch, crop_shape, padding=None):
	batch=batch.reshape((batch.shape[0], int(np.sqrt(batch.shape[1])), int(np.sqrt(batch.shape[1]))))
	oshape = np.shape(batch[0])
	if padding:
	    oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
	new_batch = []
	npad = ((padding, padding), (padding, padding))
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

def aug_data_set(ori_data, ori_labels, times_expand=1, aug_type="crop"):
	aug_data_list = []
	new_data=ori_data
	new_label=ori_labels
	for time_aug in range(times_expand):
		if aug_type == "crop": 
			crop_data = random_crop(ori_data, crop_shape=(8, 8), padding=1)
		elif aug_type == "flip_up_down":
			crop_data = flip_up_down(ori_data)
		aug_data_list.append(crop_data.reshape(crop_data.shape[0], int(crop_data.shape[1]**2)))
		new_data = np.concatenate((new_data,aug_data_list[time_aug]),axis=0)
		new_label = np.concatenate((new_label,ori_labels), axis=0)
	return new_data, new_label

def expand_features(data):
	expanded_data = []
	for idx, item in enumerate(data):
		tmp_feature = item
		for feat_idx in range(len(item)-1):
			elem = item[feat_idx]
			for i in range(feat_idx, len(item)):
				if feat_idx != i:
					tmp_feature = np.append(tmp_feature, elem*item[i])
		expanded_data.append(tmp_feature)
#	print(expanded_data[0].shape)
	return np.array(expanded_data) 

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

if __name__ == "__main__":
	digits = load_digits()
	std_images = whitening_image(digits.images)
	raw_data = std_images.reshape((std_images.shape[0], std_images.shape[1]*std_images.shape[2]))
	expanded_data = expand_features(raw_data)
	print(expanded_data.shape)
	exit()
#	raw_data = digits.data
	raw_labels = digits.target
	train_data, test_data, train_labels, test_labels = train_test_split(raw_data, raw_labels, test_size=0.2, random_state=42)
	train_set_binary, train_labels_binary, test_set_binary, test_labels_binary = extract_for_binary(
		train_data, train_labels, test_data, test_labels)
#	sample_interval = [i for i in range(5, 280, 10)]
	aug_interval = [j for j in range(1, 50, 2)]
	sample_data, sample_labels = down_sample(train_set_binary, train_labels_binary, down_sample_num=5)
	# for images with label of 6 or 8, there are 290 in total
	# for images with label of 8 or 9, there are 288 in total
	# print(train_set_binary.shape, test_set_binary.shape)
	acc_list = []
	raw_model = LogisticRegression(max_iter=500)
	raw_model.fit(sample_data, sample_labels)
	raw_accuracy = raw_model.score(test_set_binary, test_labels_binary)
	print(raw_accuracy)
	for j in aug_interval:
#		sample_data, sample_labels = down_sample(train_set_binary, train_labels_binary, down_sample_num=i)	
		new_data, new_labels = aug_data_set(sample_data, sample_labels, times_expand=j)
		print(new_data.shape)
		new_model = LogisticRegression(max_iter=500)
		new_model.fit(new_data, new_labels)
		new_accuracy = new_model.score(test_set_binary, test_labels_binary)
		acc_list.append(new_accuracy)
	print(acc_list)

