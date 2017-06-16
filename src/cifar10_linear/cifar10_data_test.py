import data_preparation.cifar10_input as cifar10

if __name__ == "__main__":
	cifar10.maybe_download_and_extract()
	train_data, train_label = cifar10.prepare_train_data(0)
	test_data, test_label = cifar10.read_validation_data()
	train_set = train_data.reshape((train_data.shape[0], train_data.shape[1]*train_data.shape[2]*train_data.shape[3]))
	test_set = test_data.reshape((test_data.shape[0], test_data.shape[1]*test_data.shape[2]*test_data.shape[3]))
	print(train_set.shape, test_set.shape)