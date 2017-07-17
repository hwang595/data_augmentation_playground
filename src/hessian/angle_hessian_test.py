import math

import numdifftools as nd
from loss_sklearn_modified import log_loss
import numpy as np

import matplotlib.pyplot as plt

def find_hyperplane_vector(angle=None):
	sup_angle = math.pi/2+angle
	norm_val = np.linalg.norm([math.sin(sup_angle), math.cos(sup_angle)])
	return np.array([[math.cos(sup_angle)/norm_val],[math.sin(sup_angle)/norm_val]])

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
	x = np.array([[0, 2, 1], [2, 0, 0]])
	for idx, data_vec in enumerate(x):
		tmp_loss += math.log(1+math.exp(-data_vec[-1]*np.dot(np.transpose(w),data_vec[0:-1])))
	return 1/float(x.shape[0])*tmp_loss

def cosh(x):
    return (math.exp(x) + math.exp(-x)) / 2

def logcosh(w):
	'''a new kind of loss here'''
	tmp_loss = 0
	X = np.array([[0, 2, 1], [2, 0, 0]])
	for idx, data_vec in enumerate(X):
		tmp_loss += math.log(cosh(np.dot(np.transpose(w),data_vec[0:-1])-data_vec[-1]))
	return 1/float(X.shape[0])*tmp_loss

if __name__ == "__main__":
	w = find_hyperplane_vector(angle=2*math.pi/8)
	H = nd.Hessian(logcosh)([float(w[0]), float(w[1])])
	print(np.linalg.eig(H)[0])
	#H = nd.Hessian(logcosh)([float(w[0]), float(w[1])])

	interval_0 = [math.pi/10, math.pi/9, math.pi/8, math.pi/7, math.pi/6, math.pi/5, math.pi/4, math.pi/3, 2*math.pi/5,3*math.pi/7]
	largest_eig_sigmoid = []
	for angle in interval_0:
		w = find_hyperplane_vector(angle=angle)
		loss_val = logistic_loss(w)
		H = nd.Hessian(logistic_loss)([float(w[0]), float(w[1])])
		largest_eig_sigmoid.append(np.amax(np.linalg.eig(H)[0]))

	interval_1 = [math.pi/10, math.pi/9, math.pi/8, math.pi/7, math.pi/6, math.pi/5, math.pi/4, 2*math.pi/5]
	largest_eig_crossentropy = []
	for angle in interval_1:
		w = find_hyperplane_vector(angle=angle)
		H = nd.Hessian(log_loss)([float(w[0]), float(w[1])])
		largest_eig_crossentropy.append(np.amax(np.linalg.eig(H)[0]))

	interval_2 = [math.pi/10, math.pi/9, math.pi/8, math.pi/7, math.pi/6, math.pi/5, math.pi/4, math.pi/3, 2*math.pi/5,3*math.pi/7]
	largest_eig_logcosh = []
	for angle in interval_2:
		w = find_hyperplane_vector(angle=angle)
		H = nd.Hessian(logcosh)([float(w[0]), float(w[1])])
		print(np.linalg.eig(H)[0])
		print('==========================================================')
		largest_eig_logcosh.append(np.amax(np.linalg.eig(H)[0]))
	plt.figure()
	plt.plot(interval_0, largest_eig_sigmoid, '^-r', label="logistic regression loss")
	plt.plot(interval_1, largest_eig_crossentropy, 's-g', label='cross entropy loss')
	plt.plot(interval_2, largest_eig_logcosh, 'v-b', label='logcosh loss')
	plt.ylim([0, 3])
	plt.legend()
	plt.xlabel("angle in rad")
	plt.ylabel("max eigen val in Hessian matrix")
	plt.title("hyperplane angle vs max Hessian eig val")
	plt.show()
