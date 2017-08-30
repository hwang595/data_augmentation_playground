import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DIST_ = 20

def rand_point_generator(point_num=None):
	'''
	we want y \in [1.5, 2.5], x \in [-0.5 0.5] for datapoints with label 1
	we want y \in [-0.5, 0.5], x \in [1.5 2.5] for datapoints with label -1
	return:
	point_num data points with label 1, point_num data points with label -1
	'''
	pos_data_points = []
	neg_data_points = []
	while len(pos_data_points) < point_num or len(neg_data_points) < point_num:
		# first settings

		x_pos_ = np.random.randint(low=-1000, high=-100) / float(1000) * DIST_
		y_pos_ = np.random.randint(low=600, high=1400) / float(1000) * DIST_
		x_neg_ = np.random.randint(low=500, high=1500) / float(1000) * DIST_
		y_neg_ = np.random.randint(low=-1000, high=-200) / float(1000) * DIST_

		# second settings shift very far
		'''
		x_pos_ = np.random.randint(low=-1000, high=-200) / float(1000)
		y_pos_ = np.random.randint(low=50000, high=51000) / float(1000)
		x_neg_ = np.random.randint(low=29000, high=31000) / float(1000)
		y_neg_ = np.random.randint(low=-5000, high=-4000) / float(1000)
		'''
		if [x_pos_, y_pos_] not in pos_data_points:
			pos_data_points.append([x_pos_, y_pos_, 1])
		if [x_neg_, y_neg_] not in neg_data_points:
			neg_data_points.append([x_neg_, y_neg_, -1])
	return np.array(pos_data_points), np.array(neg_data_points)

def find_point_with_distance(center_point_0=None, center_point_1=None, distance=None):
	# find normalized direction vector between center0 and center1
	v_ = (center_point_1 - center_point_0) / float(np.linalg.norm(center_point_1 - center_point_0))
	return center_point_0 + distance * v_

def rand_point_generator_high_dim(point_num=None, dim=None, dist=None):
	'''
	param: point_num: num of data points we want for both pos and neg dataset
	param: dim: in what dimension the data points in
	param: dist: how far away we want the two data points
	'''
	np.random.seed(seed=42)
	POS_HIGH_ = -200
	POS_LOW_  = -1200
	NEG_HIGH_ = 1800
	NEG_LOW_ = 400
	sigma_ = 0.1
	pos_data_points = []
	neg_data_points = []
	pos_labels = []
	neg_labels = []
	tmp_pos_ = np.zeros(dim)
	tmp_neg_ = np.zeros(dim)
	# we randomly generate two data points first, then based on them, we further generate more
	# data points
	for i in range(dim):
		tmp_pos_[i] = np.random.randint(low=POS_LOW_, high=POS_HIGH_) / float(1000)
		tmp_neg_[i] = np.random.randint(low=NEG_LOW_, high=NEG_HIGH_) / float(1000)

	# we generate another center by one center and distance predefined
	while len(pos_data_points) < point_num or len(neg_data_points) < point_num:
		pos_data_point = np.zeros(dim)
		neg_data_point = np.zeros(dim)
		for i in range(dim):
			pos_data_point[i] = np.random.randint(low=POS_LOW_, high=POS_HIGH_) / float(1000) * dist
			neg_data_point[i] = np.random.randint(low=NEG_LOW_, high=NEG_HIGH_) / float(1000) * dist
			pos_data_points.append(pos_data_point)
			neg_data_points.append(neg_data_point)
			pos_labels.append(1)
			neg_labels.append(-1)
	'''
	pos = tmp_pos_
	new_neg = find_point_with_distance(tmp_pos_, tmp_neg_, distance=dist)

	while len(pos_data_points) < point_num or len(neg_data_points) < point_num:
		pos_data_point = np.zeros(dim)
		neg_data_point = np.zeros(dim)
		for i in range(dim):
			pos_data_point[i] = np.random.normal(pos[i], sigma_)
			neg_data_point[i] = np.random.normal(new_neg[i], sigma_)
		pos_data_points.append(pos_data_point)
		neg_data_points.append(neg_data_point)
		pos_labels.append(1)
		neg_labels.append(-1)
	'''
	return np.array(pos_data_points), np.array(neg_data_points), np.array(pos_labels), np.array(neg_labels)


def get_transformation(angle=None):
	'''
	angles determined here is in anti-clockwise
	'''
	theta = np.radians(angle)
	c, s = np.cos(theta), np.sin(theta)
	R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
	return np.array(R)

if __name__ == "__main__":
	np.random.seed(seed=42)
	X_pos, X_neg, y_pos, y_neg = rand_point_generator_high_dim(point_num=50, dim=6, dist=0.5)
	X = np.concatenate((X_pos, X_neg), axis=0)
	#plt.show()
	'''
	pca_pos = PCA(n_components=2)
	pca_neg = PCA(n_components=2)
	X_decomp_pos=pca_pos.fit_transform(X_pos)
	X_decomp_neg=pca_neg.fit_transform(X_neg)
	'''
	pca = PCA(n_components=2)
	X_decomp = pca.fit_transform(X)
#	fig = plt.figure()
#	ax = fig.add_subplot(111, projection='3d')
#	ax.scatter(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], c='r', marker='^')
#	ax.scatter(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], c='b', marker='s')
#	plt.show()
	#print(X_decomp_pos.shape)
	#print(X_decomp_neg.shape)

	plt.figure(2)
	plt.hold(True)
	for i in range(X_decomp.shape[0]):
		if i < X_decomp.shape[0] / 2:
			plt.plot(X_decomp[i, 0], X_decomp[i, 1], '^r')
		else:
			plt.plot(X_decomp[i, 0], X_decomp[i, 1], '^b')
	#plt.plot(X_decomp_neg[:, 0], X_decomp_neg[:, 1], 'sb')
	plt.show()

	#print(np.linalg.norm(tmp_pos-new_neg))
	#print(tmp_pos.shape)
	#print(new_neg.shape)
	'''
	pos_data_points, neg_data_points=rand_point_generator(point_num=50)
	dataset = np.concatenate((pos_data_points, neg_data_points), axis=0)
	rotation_matrix = get_transformation(angle=60)
	pos_transformed = np.dot(pos_data_points[:,0:2], rotation_matrix)
	neg_transformed = np.dot(neg_data_points[:,0:2], rotation_matrix)

	fig = plt.figure(1)
	plt.scatter([x[0] for x in pos_data_points], [x[1] for x in pos_data_points], c='r')
	plt.scatter([x[0] for x in neg_data_points], [x[1] for x in neg_data_points], c='b')

	#fig_2 = plt.figure(2)
	plt.scatter([x[0] for x in pos_transformed], [x[1] for x in pos_transformed], c='r', marker='^')
	plt.scatter([x[0] for x in neg_transformed], [x[1] for x in neg_transformed], c='b', marker='^')	
	plt.show()
	'''
		


