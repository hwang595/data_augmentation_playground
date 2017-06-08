from data_prep.get_data import prepare_2D_dataset, prepare_aug_dataset, rand_data_generator, get_fliped_dataset
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

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
	plt.figure()
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, alpha=0.4)
	plt.scatter(X[:, 0], X[:, 1], c=color_tmp, alpha=0.8)
	

if __name__ == "__main__":
	# make comparison before and after data aug on perceptron
	# train on normal data

	#train_data, train_label = prepare_2D_dataset()
	# with no regularization
	train_data, train_label = rand_data_generator()
	perceptron_classifier = Perceptron()
	for i in range(len(train_data)):
		print train_data[i], train_label[i]
	perceptron_classifier.fit(train_data, train_label)
	plot_decision_boundary(clf=perceptron_classifier,
							train_data=train_data,
							train_label=train_label)
	# do some regularization(l2)
	perceptron_classifier_l2 = Perceptron(penalty='l2', alpha=0.0001)
	perceptron_classifier_l2.fit(train_data, train_label)
	plot_decision_boundary(clf=perceptron_classifier_l2,
							train_data=train_data,
							train_label=train_label)
	# do some regularization(l1)
	perceptron_classifier_l1 = Perceptron(penalty='l1', alpha=0.0001)
	perceptron_classifier_l1.fit(train_data, train_label)
	plot_decision_boundary(clf=perceptron_classifier_l1,
							train_data=train_data,
							train_label=train_label)

	# train on aug data
	#aug_train_data, aug_train_label = prepare_aug_dataset()
	aug_train_data, aug_train_label = prepare_aug_dataset(train_data, train_label, is_dummy=True)
	perceptron_classifier_aug = Perceptron()
	perceptron_classifier_aug.fit(aug_train_data, aug_train_label)
	plot_decision_boundary(clf=perceptron_classifier_aug,
							train_data=aug_train_data,
							train_label=aug_train_label)

	# train on aug data
	#aug_train_data, aug_train_label = prepare_aug_dataset()
	rand_data_fp, rand_label_fp = get_fliped_dataset(data_set=train_data)
	perceptron_classifier_fp = Perceptron()
	perceptron_classifier_fp.fit(rand_data_fp, rand_label_fp)
	plot_decision_boundary(clf=perceptron_classifier_aug,
							train_data=rand_data_fp,
							train_label=rand_label_fp)

	plt.show()

