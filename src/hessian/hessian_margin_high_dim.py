import math
import argparse

import numdifftools as nd
from loss_sklearn_modified import log_loss
from random_sample_points import rand_point_generator_high_dim, get_transformation, rand_point_generator
import numpy as np
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

SEED_ = 42
DIST_ = 8
DIM_ = 1

def find_hyperplane_vector(angle=None):
	sup_angle = math.pi/2+angle
	norm_val = np.linalg.norm([math.sin(sup_angle), math.cos(sup_angle)])
	return np.array([[math.cos(sup_angle)/norm_val],[math.sin(sup_angle)/norm_val]]), np.array([[math.cos(angle)/norm_val],[math.sin(angle)/norm_val]])

def globale_val_setter(val, val_d):
	global DIST_, DIM_
	DIST_ = val
	DIM_ = val_d

def normalize_vector(w):
	'''
	param: w: parameter of hyperplane, numpy ndarray
	'''
	dim = w.shape[1]
	normalized_w = np.zeros(dim)
	w_norm = np.linalg.norm(w)
	for elem_idx, elem in enumerate(w[0]):
		normalized_w[elem_idx] = elem / w_norm
	return normalized_w

def logistic_loss(w):
	'''x is the fake dataset wrapped with labels
		in this quite simple example we just have two data points
		in our set x

		input:
		x = np.array([0, 2, 1], [2, 0, 0])

		output:
		loss values
		'''
	tmp_loss = 0
	np.random.seed(seed=SEED_)
	#pos_data_points, neg_data_points=rand_point_generator(point_num=50)
	X_pos, X_neg, y_pos, y_neg = rand_point_generator_high_dim(point_num=50, dim=DIM_, dist=DIST_)
	#pos_data_points, neg_data_points=rand_point_generator(50)
	#x = np.concatenate((pos_data_points[:,0:-1], neg_data_points[:,0:-1]), axis=0)
	#y = np.concatenate((pos_data_points[:,-1], neg_data_points[:,-1]), axis=0)
	x = np.concatenate((X_pos, X_neg), axis=0)
	y = np.concatenate((y_pos, y_neg), axis=0)

	for idx, data_vec in enumerate(x):
		#tmp_loss += math.log(1+math.exp(-data_vec[-1]*np.dot(np.transpose(w),data_vec[0:-1])))
		tmp_loss += math.log(1+math.exp(-y[idx]*np.dot(np.transpose(w),data_vec)))
	return 1/float(x.shape[0])*tmp_loss

def fetch_margin(dataset, w, b):
	'''
	param: w is the params of hyperplane, b is the params of intercept
	param: X is the dataset
	'''
	dist_list = []
	for i, data_point in enumerate(dataset):
		dist = abs(np.dot(data_point, np.transpose(w)) + b)
		dist_list.append(dist)
	return min(dist_list)

def cosh(x):
    return (math.exp(x) + math.exp(-x)) / 2

def logcosh(w):
	'''a new kind of loss here'''
	tmp_loss = 0
#	X = np.array([[0, 2, 1], [2, 0, 0]])
#	X = np.array([[0, 2, 1], [2, 0, -1]])
	np.random.seed(seed=SEED_)
	pos_data_points, neg_data_points=rand_point_generator(point_num=50)
	X = np.concatenate((pos_data_points, neg_data_points), axis=0)
	for idx, data_vec in enumerate(X):
		tmp_loss += math.log(cosh(np.dot(np.transpose(w),data_vec[0:-1])-data_vec[-1]))
	return 1/float(X.shape[0])*tmp_loss

def interval_generator(start=None, end=None, interval=None):
	tmp_interval = []
	cur_val = start
	while cur_val <= end:
		tmp_interval.append(cur_val)
		cur_val += interval
	if end not in tmp_interval:
		tmp_interval.append(end)
	return tmp_interval

def generate_graph(w_vec_list=None):
	interval_for_plot = []

	np.random.seed(seed=SEED_)
	pos_data_points, neg_data_points=rand_point_generator(point_num=50)
	fig = plt.figure(1)
	plt.scatter([x[0] for x in pos_data_points], [x[1] for x in pos_data_points],  c='r')
	plt.scatter([x[0] for x in neg_data_points], [x[1] for x in neg_data_points],  c='b')
	interval_for_plot = np.arange(-2, 50)
	for vec_idx_, w_ in enumerate(w_vec_list):
		x_table_tmp = []
		y_table_tmp = []
		for points in interval_for_plot:
			x_table_tmp.append(points*w_[0])
			y_table_tmp.append(points*w_[1])
		plt.plot(x_table_tmp, y_table_tmp)
	plt.show()


if __name__ == "__main__":
	np.random.seed(seed=SEED_)
	dist_factor_range_list = [i for i in range(1, 15)]
	dim_list = [i for i in range(3, 21, 2)]
	margin_list = []
	trace_list = []
	dist_factor_range_list.insert(0, 0.5)
	for d in dim_list:
		margin_list_tmp = []
		trace_list_tmp = []
		for f in dist_factor_range_list:
			# set the value of DIST_ here
			globale_val_setter(f, d)
			X_pos, X_neg, y_pos, y_neg = rand_point_generator_high_dim(point_num=50, dim=DIM_, dist=DIST_)
			#pos_data_points, neg_data_points=rand_point_generator(50)
			#X = np.concatenate((pos_data_points[:,0:-1], neg_data_points[:,0:-1]), axis=0)
			#y = np.concatenate((pos_data_points[:,-1], neg_data_points[:,-1]), axis=0)

			X = np.concatenate((X_pos, X_neg), axis=0)
			y = np.concatenate((y_pos, y_neg), axis=0)

			SVM_classifier = LinearSVC(C=1e10, fit_intercept=False)
			SVM_classifier.fit(X, y)

			if f == 0.5:
				print(SVM_classifier.score(X, y))
				print('------------------------------------------------------')

			w = SVM_classifier.coef_
			normalized_w = normalize_vector(w)
			b = SVM_classifier.intercept_
			margin=fetch_margin(X, normalized_w, b)

			#fig = plt.figure(1)
			#plt.plot(X_pos[:,0], X_pos[:,1], '^r')
			#plt.plot(X_neg[:,0], X_neg[:,1], '^b')
			#plt.show()

	#		print(margin)
	#		print
			margin_list_tmp.append(margin)

			H = nd.Hessian(logistic_loss)([float(normalized_w[i]) for i in range(normalized_w.shape[0])])
			trace_list_tmp.append(np.amax(np.linalg.eig(H)[0]))
			#trace_list_tmp.append(np.trace(H))
		print("Dim {dim} is done".format(dim = d))
		margin_list.append(margin_list_tmp)
		trace_list.append(trace_list_tmp)
		#H = nd.Hessian(logistic_loss)([float(w[0][i]) for i in range(len(w[0]))])
		#largest_eig_sigmoid.append(np.amax(np.linalg.eig(H)[0]))
#		print(np.amax(np.linalg.eig(H)[0]))

	plt.figure(1)
	assert len(margin_list) == len(trace_list)
	for i in range(len(margin_list)):
		plt.plot(margin_list[i], trace_list[i], label="dimension: {i}".format(i=dim_list[i]))
	plt.xlabel("margin")
	plt.ylabel("trace of Hessian matrix")
	plt.legend()
	plt.show()

	'''
	data_pos, data_neg=rand_point_generator(point_num=50)
	X_pos = data_pos[:, 0:-1]
	X_neg = data_neg[:, 0:-1]
	y_pos = data_pos[:, -1]
	y_neg = data_neg[:, -1]
	X = np.concatenate((X_pos, X_neg), axis=0)
	y = np.concatenate((y_pos, y_neg), axis=0)

	SVM_classifier = LinearSVC(C=1e10, fit_intercept=True)
	SVM_classifier.fit(X, y)
	w = SVM_classifier.coef_

	normalized_w = normalize_vector(w)
	b = SVM_classifier.intercept_

	fetch_margin(X, normalized_w, b)
	'''
	