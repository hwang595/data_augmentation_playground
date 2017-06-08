from data_prep.get_data import prepare_2D_dataset, prepare_aug_dataset, rand_data_generator, get_fliped_dataset
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
#	train_data, train_label = prepare_2D_dataset()
#	X_train, X_test, y_train, y_test=train_test_split(train_data, train_label, test_size=0.2, random_state=42)
	train_data, train_label = rand_data_generator()
	perceptron_classifier = SVC(C=1)
	perceptron_classifier.fit(train_data, train_label)
#	svc_pred=perceptron_classifier.predict(X_test)
#	gen_err=accuracy_score(y_test, svc_pred)

	plot_decision_boundary(clf=perceptron_classifier,
							train_data=train_data,
							train_label=train_label)

	# train on aug data
#	aug_train_data, aug_train_label = prepare_aug_dataset(is_dummy=False)
#	X_train_aug, X_test_aug, y_train_aug, y_test_aug=train_test_split(aug_train_data, aug_train_label, test_size=0.2, random_state=42)
	aug_train_data, aug_train_label = prepare_aug_dataset(train_data, train_label, is_dummy=True)
	perceptron_classifier_aug = SVC(C=1)
	perceptron_classifier_aug.fit(aug_train_data, aug_train_label)
#	svc_aug_pred=perceptron_classifier.predict(X_test_aug)
#	gen_aug_err=accuracy_score(y_test_aug, svc_aug_pred)
#	print gen_err, gen_aug_err
	plot_decision_boundary(clf=perceptron_classifier_aug,
							train_data=aug_train_data,
							train_label=aug_train_label)

	# try flip the data up and down
	train_data_fp, train_label_fp = get_fliped_dataset(train_data)
	svc_fp = SVC(C=1)
	svc_fp.fit(train_data_fp, train_label_fp)
#	svc_aug_pred=perceptron_classifier.predict(X_test_aug)
#	gen_aug_err=accuracy_score(y_test_aug, svc_aug_pred)
#	print gen_err, gen_aug_err
	plot_decision_boundary(clf=svc_fp,
							train_data=train_data_fp,
							train_label=train_label_fp)
	plt.show()