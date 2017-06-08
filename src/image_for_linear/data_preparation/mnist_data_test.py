'''this is to verify how the data works'''
import mnist_data

if __name__ == "__main__":
	dataset = mnist_data.load_mnist(reshape=True)
	dataset.train.test_print
	#print(dataset.train.num_examples, dataset.train.data_point_shape)
	#print(dataset.validation.num_examples, dataset.validation.data_point_shape)