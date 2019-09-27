
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load(filename):		
	data = arff.loadarff(filename)
	df = pd.DataFrame(data[0])
	# Convert string attribute to integer
	df.outlook = pd.Categorical(pd.factorize(df.outlook)[0])
	df.outlook = pd.to_numeric(df.outlook, errors='coerce')
	df.windy = pd.Categorical(pd.factorize(df.windy)[0])
	df.windy = pd.to_numeric(df.windy, errors='coerce')
	df.play = pd.Categorical(pd.factorize(df.play)[0])
	df.play = pd.to_numeric(df.play, errors='coerce')

	df.head()
	return df

def split(data, test_size=0.2):
	y = data.play
	x = data.drop('play',axis=1)
	scaler = preprocessing.MinMaxScaler().fit(x)
	x = scaler.transform(x)
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	return X_train, X_test, y_train, y_test

class FFNN: 
	def __init__(self, nb_feature, nb_hidden_layer, nb_nodes):
		self.nb_output_layer = 1
		self.nb_feature = nb_feature
		self.nb_hidden_layer = nb_hidden_layer
		self.nb_nodes = nb_nodes
		self.nb_weight_layer = nb_hidden_layer + 1
		
		self.weights = []
		self.output_neurons = []
		self.feature_neurons = []
		self.hidden_neurons = []

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x)) 

	def count_error(self, target_output):
		return pow(self.learning_rate - target_output)/2
	
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
	

	def get_sigmoid_value(self, m=None, n=None, output=False):
		if output:
			m = self.nb_weight_layer-1
			n = 0
		previous_layer = self.feature_neurons if m == 0 else self.hidden_neurons[m-1]
		weight_layer = self.weights[m]
		sigmoid_value = 0
		
		for i in range(len(previous_layer)):
			sigmoid_value += previous_layer[i] * weight_layer[i, n]
		sigmoid_value = self.sigmoid(sigmoid_value)
		return sigmoid_value

	def get_output(self, input):
		self.feature_neurons = []
		for item in input:
			self.feature_neurons.append(item)

		for i in range(self.nb_hidden_layer):
			temp_hidden_layer = []
			for j in range(self.nb_nodes):
				temp_hidden_layer.append(self.get_sigmoid_value(i, j))
			self.hidden_neurons.append(temp_hidden_layer)
		output = self.get_sigmoid_value(output=True)	
		return output	

	def fit(self, x_train, y_train, batch_size, momentum=0.001, learning_rate=0.5, epoch=5):
		self.init_weights()
		for i in range(epoch):
			#masuk ke epoch
			for	j in range(0, len(y_train), batch_size):
				#masuk ke batch
				x_mini = x_train[j:j+batch_size]
				y_mini = y_train[j:j+batch_size]
				x_output = []
				for x in x_mini:
					x_output.append(self.get_output(x))
				
				delta_x = []
				# for x, y in x_mini, y_mini:
					# itung selisih prediksi dgn ekspektasi


if __name__ == "__main__":
	data = load('weather.arff')
	X_train, X_test, y_train, y_test = split(data, 0.2)
	ffnn = FFNN(X_train.shape[1],2,3)
	ffnn.fit(X_train, y_train, 2)
        