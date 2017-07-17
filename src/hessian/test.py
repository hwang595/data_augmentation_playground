from loss_sklearn_modified import log_loss

if __name__ == "__main__":
	a = log_loss([0, 1], [1, 0])
	print(a)