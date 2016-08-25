import numpy as np

class neuralNetwork:
	def __init__(self, input_layers, hidden_layers, output_layers, alpha):
		self.inodes = input_layers
		self.hnodes = hidden_layers
		self.onodes = output_layers
		
		self.syn0 = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.syn1 = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		self.alpha = alpha
		self.activationFunction = lambda x: self.sigmoid(x)
		pass

	def sigmoid(self, x):
		return 1 / ( 1 + np.exp(-x) )

	def sigmoid_derivative(self, output):
		return output * (1 - output)

	def feedForward(self, inputData):
		l0 = np.array(inputData, ndmin=2).T
		l1 = self.activationFunction(np.dot(self.syn0, l0))
		l2 = self.activationFunction(np.dot(self.syn1, l1))
		return (l0, l1, l2)

	def train(self, inputData, targetsList):
		y = np.array(targetsList, ndmin=2).T
		(l0, l1, l2) = self.feedForward(inputData)

		l2_error = y - l2
		l2_delta = l2_error * self.sigmoid_derivative(l2)

		l1_error = np.dot(self.syn1.T, l2_delta)
		l1_delta = l1_error * self.sigmoid_derivative(l1)

		self.syn1 += self.alpha * np.dot(l2_delta, l1.T)
		self.syn0 += self.alpha * np.dot(l1_delta, l0.T)
		pass

	def query(self, inputData):
		(l0, l1, l2) = self.feedForward(inputData)
		return l2

inputNodes = 784
hiddenNodes = 100
outputNodes = 10

learningRate = .3
n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

training_data_file = open("mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

i = 0
for record in training_data_list:
	all_values = record.split(',')
	inputs = (np.asfarray(all_values[1:]) / 255.0 * .99) + 0.01
	targets = np.zeros(outputNodes) + .01
	targets[int(all_values[0])] = .99
	n.train(inputs, targets)
	if i % 1000 == 0:
		print(i)
	i += 1
	pass

# Testing
# This will display the id of the image, followed by an array of 10 floats,
# displaying the probability of each number
for j in range(100):
	all_values = training_data_list[j].split(',')
	print(all_values[0])
	print(n.query((np.asfarray(all_values[1:]) / 255.0 * .99) + 0.01))





