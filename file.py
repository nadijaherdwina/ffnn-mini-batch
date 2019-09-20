
import numpy as np


# class Neuron:
# 	def __init__(self, value):
# 		self.value = value
# 		self.weight = weight

class FFNN: 
	def __init__(self, nb_feature, nb_hidden_layer, nb_nodes, batch_size, epoch=10, momentum=0.0001, learning_rate=0.5):
		self.nb_output_layer = 1
		self.nb_feature = nb_feature
		self.nb_hidden_layer = nb_hidden_layer
		self.nb_nodes = nb_nodes
		self.nb_weight_layer = nb_hidden_layer + 1
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.batch_size = batch_size
		self.epoch = epoch
		
		self.weights = []
		self.output_neurons = []
		self.feature_neurons = []
		self.hidden_neurons = []

	def softmax(self, x):
		return 0 if x < 0 else x

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x)) 

	def count_error(self, target_output):
		return pow(self.learning_rate - target_output)/2

	def update_weight(self):
		return 0
		
	def init_weights(self):
		for i in range (self.nb_weight_layer):
			temp_weight = []
			if i == 0:
				temp_weight = np.random.randn(self.nb_feature, self.nb_nodes)
			elif i < self.nb_weight_layer - 1:
				temp_weight = np.random.randn(self.nb_nodes, self.nb_nodes)
			elif i == self.nb_weight_layer - 1:
				temp_weight = np.random.randn(self.nb_nodes, self.nb_output_layer)
			self.weights.append(temp_weight)
	
	def init_hidden_neurons(self):
		#init array hidden, diisi 0 aja yang penting kebikin
		for i in range(self.nb_hidden_layer):
			temp_hidden_layer = []
			for i in range(self.nb_nodes):
				temp_hidden_layer.append(0)
			self.hidden_neurons.append(temp_hidden_layer)

	def init_feature_neurons(self):
		# harusnya ambil nilai tiap fitur dari row yang lagi diitung
		for i in range (self.nb_feature):
			self.feature_neurons.append(0)
	
	def make_network(self):
		self.init_weights()
		self.init_feature_neurons()
		self.init_hidden_neurons()

	def get_sigmoid_value(self, m=None, n=None, output=False):
		
		if output:
			m = self.nb_weight_layer-1
			n = 0
	
		previous_layer = self.feature_neurons if m == 0 else self.hidden_neurons[m-1]
		weight_layer = self.weights[m]
		sigmoid_value = 0

		for i in range(len(previous_layer)):
			sigmoid_value += previous_layer[i] * weight_layer[i, n]
			print(m,i,n)
		sigmoid_value = self.sigmoid(sigmoid_value)
		# print('sigmoid_value= sigmoid(w',m, i, n, ' * h', m-1,i ,')')
		return sigmoid_value


if __name__ == "__main__":
	ffnn = FFNN(2,2,3,5)
	ffnn.make_network()
	ffnn.get_sigmoid_value(output=True)
        