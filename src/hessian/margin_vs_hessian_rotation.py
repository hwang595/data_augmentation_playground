import math
import argparse

import numdifftools as nd
from loss_sklearn_modified import log_loss
from random_sample_points import rand_point_generator, get_transformation
import numpy as np

import matplotlib.pyplot as plt

SEED_ = 42
ANGLE_ = -45

def find_hyperplane_vector(angle=None):
	sup_angle = math.pi/2+angle
	norm_val = np.linalg.norm([math.sin(sup_angle), math.cos(sup_angle)])
	return np.array([[math.cos(sup_angle)/norm_val],[math.sin(sup_angle)/norm_val]]), np.array([[math.cos(angle)/norm_val],[math.sin(angle)/norm_val]])

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
#	x = np.array([[0, 2, 1], [2, 0, 0]])
#	x = np.array([[0, 2, 1], [2, 0, -1]])
#	x = np.array([[-0.5, 1, 1], [0.7, -0.5, -1]])
	np.random.seed(seed=SEED_)
	pos_data_points, neg_data_points=rand_point_generator(point_num=50)

	rotation_matrix = get_transformation(angle=ANGLE_)
	pos_transformed = np.dot(pos_data_points[:,0:2], rotation_matrix)
	neg_transformed = np.dot(neg_data_points[:,0:2], rotation_matrix)
	x = np.concatenate((pos_transformed, neg_transformed), axis=0)
	x_new = np.zeros((x.shape[0], 3))
	for idx_x_p, x_p in enumerate(x):
		if idx_x_p <= 49:
			x_new[idx_x_p] = np.append(x_p, 1)
		else:
			x_new[idx_x_p] = np.append(x_p, -1)
	#x_ = np.concatenate((pos_data_points, neg_data_points), axis=0)
	#y_ = x[:,-1]

	for idx, data_vec in enumerate(x_new):
		tmp_loss += math.log(1+math.exp(-data_vec[-1]*np.dot(np.transpose(w),data_vec[0:-1])))
#		tmp_loss += math.log(1+math.exp(-y_[idx]*np.dot(np.transpose(w),data_vec)))
	return 1/float(x.shape[0])*tmp_loss

def cosh(x):
    return (math.exp(x) + math.exp(-x)) / 2

def logcosh(w):
	'''a new kind of loss here'''
	tmp_loss = 0
#	X = np.array([[0, 2, 1], [2, 0, 0]])
#	X = np.array([[0, 2, 1], [2, 0, -1]])
	np.random.seed(seed=SEED_)
	pos_data_points, neg_data_points=rand_point_generator(point_num=50)
#	X = np.concatenate((pos_data_points, neg_data_points), axis=0)

	rotation_matrix = get_transformation(angle=ANGLE_)
	pos_transformed = np.dot(pos_data_points[:,0:2], rotation_matrix)
	neg_transformed = np.dot(neg_data_points[:,0:2], rotation_matrix)
	X = np.concatenate((pos_transformed, neg_transformed), axis=0)
	X_new = np.zeros((X.shape[0], 3))
	for idx_x_p, x_p in enumerate(X):
		if idx_x_p <= 49:
			X_new[idx_x_p] = np.append(x_p, 1)
		else:
			X_new[idx_x_p] = np.append(x_p, -1)

	for idx, data_vec in enumerate(X_new):
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

	rotation_matrix = get_transformation(angle=ANGLE_)
	pos_transformed = np.dot(pos_data_points[:,0:2], rotation_matrix)
	neg_transformed = np.dot(neg_data_points[:,0:2], rotation_matrix)
	
	fig = plt.figure(1)
	plt.scatter([x[0] for x in pos_transformed], [x[1] for x in pos_transformed],  c='r', marker='^')
	plt.scatter([x[0] for x in neg_transformed], [x[1] for x in neg_transformed],  c='b', marker='^')

#	plt.scatter([x[0] for x in pos_data_points], [x[1] for x in pos_data_points],  c='r')
#	plt.scatter([x[0] for x in neg_data_points], [x[1] for x in neg_data_points],  c='b')
	interval_for_plot = np.arange(-2, 3)
	for vec_idx_, w_ in enumerate(w_vec_list):
		w_transformed = np.transpose(np.dot(np.transpose(w_), rotation_matrix))
		x_table_tmp = []
		y_table_tmp = []
		for points in interval_for_plot:
			x_table_tmp.append(points*w_transformed[0])
			y_table_tmp.append(points*w_transformed[1])
		plt.plot(x_table_tmp, y_table_tmp)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', help='mode of this program', required=True, dest='mode')
	argument = parser.parse_args()
	mode = argument.mode

	# get the random dataset:
	np.random.seed(seed=SEED_)
	pos_data_points, neg_data_points=rand_point_generator(point_num=50)
	dataset = np.concatenate((pos_data_points, neg_data_points), axis=0)
	X = dataset[:, 0:2]
	y = dataset[:, -1]
	rotation_matrix = get_transformation(angle=ANGLE_)
	X_transformed = np.dot(X, rotation_matrix)
	
	# do some test here if under debug mode
	if mode == "debug":
		w, w_fake = find_hyperplane_vector(angle=2*math.pi/8)
		w_2, w_2_fake = find_hyperplane_vector(angle=math.pi/3)
		H = nd.Hessian(logcosh)([float(w[0]), float(w[1])])
		print(np.linalg.eig(H)[0])
		interval_for_plot = []
		for i in range(-2, 3):
			interval_for_plot.append([i, 0])
		y_val_list_1 = []
		y_val_list_2 = []
		for points in interval_for_plot:
			y_val_list_1.append(np.dot(np.transpose(w), np.array(points))[0])
			y_val_list_2.append(np.dot(np.transpose(w_2), np.array(points))[0])

		fig_test = plt.figure()
		plt.plot([i for i in reversed(range(-2, 3))], y_val_list_1, '-b')
		plt.plot([i for i in reversed(range(-2, 3))], y_val_list_2, '-r')
		plt.show()
		exit()

	# save direction vectors of hyperplanes
	w_list_0 = []
	w_fake_list_0 = []
	margin_list_0 = []
	#interval_0 = interval_generator(0, math.pi/6, math.pi/100)
	interval_0 = interval_generator(0, math.pi/2, math.pi/100)
	#interval_0 = interval_generator(math.pi/4, math.pi/2, math.pi/100)
	logistic_loss_vals = []

	largest_eig_sigmoid = []
	for angle in interval_0:
		w, w_fake = find_hyperplane_vector(angle=angle)
		w = np.transpose(np.dot(np.transpose(w), rotation_matrix))
		w_list_0.append(w)
		w_fake_list_0.append(w_fake)
		loss_val = logistic_loss(w)
		logistic_loss_vals.append(loss_val)
		H = nd.Hessian(logistic_loss)([float(w[0]), float(w[1])])
		#largest_eig_sigmoid.append(np.amax(np.linalg.eig(H)[0]))
		largest_eig_sigmoid.append(np.trace(H))

	for w in w_list_0:
		margin_candidates = []
		for x_p in X_transformed:
			margin_candidates.append(abs(np.dot(np.transpose(w), x_p)))
		margin_list_0.append(min(margin_candidates))
	plt.figure(1)
	plt.subplot(221)
	plt.plot(interval_0, largest_eig_sigmoid, '^-r')
#	plt.title("angle vs max eig val")
	plt.title("angle vs trace of hessian matrix")
#	plt.xlabel("angle/rad")
#	plt.ylabel("max eig val")
	plt.ylabel("trace of hessian")
	plt.subplot(222)
	plt.title("angle vs margin")
	plt.plot(interval_0, margin_list_0, '^-g')
#	plt.xlabel("angle/rad")
	plt.ylabel("margin")	
	plt.subplot(223)
	plt.title("margin vs max eig val")
	plt.plot(margin_list_0, largest_eig_sigmoid, 's-b')
	plt.xlabel("margin")
#	plt.ylabel("max eig val")
	plt.ylabel("trace of hessian")
	plt.subplot(224)
	plt.plot(margin_list_0, logistic_loss_vals, 'v-m')
	plt.xlabel('margin')
	plt.ylabel("logistic loss")
	plt.show()

#	plt.show()
#	exit()

	w_list_1 = []
	margin_list_1 = []
	interval_1 = interval_generator(0, math.pi/2, math.pi/100)
#	interval_1 = [0, math.pi/10, math.pi/9, math.pi/8, math.pi/7, math.pi/6, math.pi/5, math.pi/4, 2*math.pi/5, math.pi/2]
	largest_eig_crossentropy = []
	for angle in interval_1:
		w, _ = find_hyperplane_vector(angle=angle)
		w = np.transpose(np.dot(np.transpose(w), rotation_matrix))
		w_list_1.append(w)
		H = nd.Hessian(log_loss)([float(w[0]), float(w[1])])
		#largest_eig_crossentropy.append(np.amax(np.linalg.eig(H)[0]))
		largest_eig_crossentropy.append(np.trace(H))
	for w in w_list_1:
		margin_candidates = []
		for x_p in X_transformed:
			margin_candidates.append(abs(np.dot(np.transpose(w), x_p)))
		margin_list_1.append(min(margin_candidates))
	plt.figure(2)
	plt.subplot(221)
	plt.plot(interval_1, largest_eig_crossentropy, '^-r')
	plt.title("angle vs trace of hessian matrix")
#	plt.ylabel("max eig val")
	plt.ylabel("trace of hessian")
	plt.subplot(222)
	plt.title("angle vs margin")
	plt.plot(interval_1, margin_list_1, '^-g')
	plt.ylabel("margin")	
	plt.subplot(223)
	plt.title("margin vs trace of hessian matrix")
	plt.plot(margin_list_1, largest_eig_crossentropy, 's-b')
	plt.xlabel("margin")
#	plt.ylabel("max eig val")
	plt.ylabel("trace of hessian")


	w_list_2 = []
	margin_list_2 = []
	interval_2 = interval_generator(0, math.pi/2, math.pi/100)
	#interval_2 = [0, math.pi/10, math.pi/9, math.pi/8, math.pi/7, math.pi/6, math.pi/5, math.pi/4, math.pi/3, 2*math.pi/5,3*math.pi/7, math.pi/2]
	largest_eig_logcosh = []
	for angle in interval_2:
		w, _ = find_hyperplane_vector(angle=angle)
		w = np.transpose(np.dot(np.transpose(w), rotation_matrix))
		w_list_2.append(w)
		H = nd.Hessian(logcosh)([float(w[0]), float(w[1])])
#		largest_eig_logcosh.append(np.amax(np.linalg.eig(H)[0]))
		largest_eig_logcosh.append(np.trace(H))
	for w in w_list_2:
		margin_candidates = []
		for x_p in X_transformed:
			margin_candidates.append(abs(np.dot(np.transpose(w), x_p)))
		margin_list_2.append(min(margin_candidates))
	plt.figure(3)
	plt.subplot(311)
	plt.plot(interval_2, largest_eig_logcosh, '^-r')
	plt.title("angle vs trace of hessian matrix")
#	plt.ylabel("max eig val")
	plt.ylabel("trace of hessian")
	plt.subplot(312)
	plt.title("angle vs margin")
	plt.plot(interval_2, margin_list_2, '^-g')
	plt.ylabel("margin")	
	plt.subplot(313)
	plt.title("margin vs trace of hessian matrix")
	plt.plot(margin_list_2, largest_eig_logcosh, 's-b')
	plt.xlabel("margin")
#	plt.ylabel("max eig val")
	plt.ylabel("trace of hessian")
	plt.show()