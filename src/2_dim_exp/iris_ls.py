# Code source: Jaques Grobler
# License: BSD 3 clause
from data_prep.get_data import prepare_2D_dataset, prepare_aug_dataset, rand_data_generator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
AA=-0.0001
BB=0.0001
def plot_decision_boundary(clf=None, train_data=None, train_label=None):
	"""plot the decision boundary for linear classifier"""
	X = train_data
	color_tmp = []
	for item in train_label:
		if item == 0:
			color_tmp.append('r')
		else:
			color_tmp.append('b')
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
	                     np.arange(y_min, y_max, 0.01))
	# Plotting decision regions
	plt.figure(1)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	#for z in Z:
	#	print z
	#exit()
	# need to reconstruct Z right here
	Z_construct = []
	for z in Z:
		tmp_z = []
		for i in z:
			if i < AA:
				tmp_z.append(0)
			elif i >= AA and i < BB:
				tmp_z.append(1)
			else:
				tmp_z.append(2)
		Z_construct.append(tmp_z)

	plt.contourf(xx, yy, Z_construct, alpha=0.4)
	plt.scatter(X[:, 0], X[:, 1], c=color_tmp, alpha=0.8)
	plt.figure(2)
	plt.contourf(xx, yy, Z, alpha=0.4)
	plt.scatter(X[:, 0], X[:, 1], c=color_tmp, alpha=0.8)

if __name__ == "__main__":
	# fetch data
	train_data, train_label = rand_data_generator()
	# Create linear regression object
	regr = linear_model.LinearRegression()
	# Train the model using the training sets
	regr.fit(train_data, train_label)
	plot_decision_boundary(regr, train_data=train_data, train_label=train_label)
	plt.show()