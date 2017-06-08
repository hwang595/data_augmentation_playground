import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

def prepare_2D_dataset():
	iris = datasets.load_iris()
	X = iris.data[:, :2]
	Y = iris.target
	return X[0:100], Y[0:100]

def do_aug_by_proj(train_set=None):
	aug_train_set = []
	for data_point_idx in range(len(train_set)):
		data_point = train_set[data_point_idx]
		aug_train_set.append(data_point)
		if data_point_idx < 50:
			aug_train_set.append(np.array([0, data_point[1]]))
		else:
			aug_train_set.append(np.array([data_point[0],0]))
	aug_train_label = [0 if i < 100 else 1 for i in range(len(aug_train_set))]
	return np.array(aug_train_set), np.array(aug_train_label)

def do_aug_by_proj_dummy(train_set=None):
	'''probably not a good one, only for test'''
	aug_train_set = []
	for data_point_idx in range(len(train_set)):
		data_point = train_set[data_point_idx]
		aug_train_set.append(data_point)
		aug_train_set.append(np.array([0, data_point[1]]))
		aug_train_set.append(np.array([data_point[0],0]))
	aug_train_label = [0 if i < len(aug_train_set)/2 else 1 for i in range(len(aug_train_set))]
	return np.array(aug_train_set), np.array(aug_train_label)

def prepare_aug_dataset(input_data=None, input_label=None,is_dummy=False):
	if input_data == None and input_label == None:
		iris = datasets.load_iris()
		X = iris.data[:, :2]
		Y = iris.target
		partial_train_set = X[0:100]
		partial_train_label = Y[0:100]
	else:
		partial_train_set = input_data
		partial_train_label = input_label
	if is_dummy:
		return do_aug_by_proj_dummy(train_set=partial_train_set)
	else:
		return do_aug_by_proj(train_set=partial_train_set)

def rand_data_generator():
	data = 2.5 * np.random.randn(2, 2) + 3
	label = np.array([0, 1])
	return data, label

def get_fliped_dataset(data_set):
	aug_train_set = []
	for data_point_idx in range(len(data_set)):
		data_point = data_set[data_point_idx]
		aug_train_set.append(data_point)
		aug_train_set.append(np.array([-data_point[0], data_point[1]]))
		aug_train_set.append(np.array([data_point[0],-data_point[1]]))
	aug_train_label = [0 if i < len(aug_train_set)/2 else 1 for i in range(len(aug_train_set))]
	return np.array(aug_train_set), np.array(aug_train_label)

if __name__ == "__main__":
	train_data, train_label = prepare_2D_dataset()
	aug_train_data, aug_train_label = prepare_aug_dataset()
	aug_train_data_dummy, aug_train_label_dummy = prepare_aug_dataset(is_dummy=True)
	rand_data, rand_label = rand_data_generator()
	rand_data_aug, rand_label_aug = prepare_aug_dataset(rand_data, rand_label, is_dummy=True)
	rand_data_fp, rand_label_fp = get_fliped_dataset(data_set=rand_data)
	'''
	plt.figure(1)
	plt.title("Iris Dataset")
	plt.subplot(131)
	plt.scatter([i[0] for i in train_data[0:50]], [j[1] for j in train_data[0:50]], c='r', alpha=0.5)
	plt.scatter([i[0] for i in train_data[50:100]], [j[1] for j in train_data[50:100]], c='b', alpha=0.5)
	plt.xlabel("original dataset")
	plt.subplot(132)
	plt.scatter([i[0] for i in aug_train_data[0:100]], [j[1] for j in aug_train_data[0:100]], c='r', alpha=0.5)
	plt.scatter([i[0] for i in aug_train_data[100:200]], [j[1] for j in aug_train_data[100:200]], c='b', alpha=0.5)
	plt.xlabel("project partially")
	plt.subplot(133)
	plt.scatter([i[0] for i in aug_train_data_dummy[0:150]], [j[1] for j in aug_train_data_dummy[0:150]], c='r', alpha=0.5)
	plt.scatter([i[0] for i in aug_train_data_dummy[150:300]], [j[1] for j in aug_train_data_dummy[150:300]], c='b', alpha=0.5)
	plt.xlabel("project onto both axis")

	plt.figure(2)
	plt.subplot(121)
	plt.scatter(rand_data[0,0],rand_data[0,1], c='r', alpha=0.5)
	plt.scatter(rand_data[1,0],rand_data[1,1], c='b', alpha=0.5)
	plt.subplot(122)
	plt.scatter(rand_data_aug[0:3,0], rand_data_aug[0:3,1],  c='r', alpha=0.5)
	plt.scatter(rand_data_aug[3:,0], rand_data_aug[3:,1], c='b',alpha=0.5)
	'''
	plt.figure(3)
	plt.subplot(131)
	plt.scatter(rand_data[0,0],rand_data[0,1], c='r', alpha=0.5)
	plt.scatter(rand_data[1,0],rand_data[1,1], c='b', alpha=0.5)
	plt.xlabel("original data points")
	plt.subplot(132)
	plt.scatter(rand_data_aug[0:3,0], rand_data_aug[0:3,1],  c='r', alpha=0.5)
	plt.scatter(rand_data_aug[3:,0], rand_data_aug[3:,1], c='b',alpha=0.5)
	plt.xlabel("augment by projection")
	plt.subplot(133)
	plt.scatter(rand_data_fp[0:3,0], rand_data_fp[0:3,1], c='r', alpha=0.5)
	plt.scatter(rand_data_fp[3:,0], rand_data_fp[3:,1], c='b', alpha=0.5)
	plt.xlabel("augment by flip")		
	plt.show()

