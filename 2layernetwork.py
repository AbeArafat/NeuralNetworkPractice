from numpy import exp, array, random, dot


class NeuronLayer():
  def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
    self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
    # #inputs arrays of #neurons numbers between -1 and 1

class NeuralNetwork():

  def __init__(self, layer1, layer2):
    self.layer1 = layer1 #create variable layer1 equal to parameter layer1 (class wide)
    self.layer2 = layer2 #create variable layer2 equal to parameter layer2 (class wide)
  

  # The Sigmoid function, which describes an S shaped curve.
  # Weighted sum is x value, returns y value
  # normalise them between 0 and 1.
  def __sigmoid(self, x):
    return 1 / (1 + exp(-x))

  # This is the gradient (slope) of the Sigmoid curve.
  # It indicates how confident we are about the existing weight.
  # Used to adjust weights
  #think, when it gets closer to |y|= 1, gradient becomes closer to |+/-1|
  def __sigmoid_derivative(self, x):
    return x * (1 - x)

  def think(self, inputs):
    output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights)) 
    #multiply inputs one subarray at a time(3x 0 to 1) by weights(matrix)(-1 to 1) and normalize values to between 0 and 1 to end with matrix same size as weights matrix
    output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))

    return output_from_layer1, output_from_layer2

  def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
      output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)
      layer2_error = training_set_outputs - output_from_layer_2
      layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
      layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
      layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)
      layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
      layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
      self.layer1.synaptic_weights += layer1_adjustment
      self.layer2.synaptic_weights += layer2_adjustment

  def print_weights(self):
    print ("    Layer 1 (4 neurons, each with 3 inputs): ")
    print (self.layer1.synaptic_weights)
    print ("    Layer 2 (1 neuron, with 4 inputs):")
    print (self.layer2.synaptic_weights)

if __name__ == "__main__":
  random.seed(1)

  # Create layer 1 (4 neurons, each with 3 inputs)
  layer1 = NeuronLayer(4, 3)
  # 3 arrays of 4 numbers between -1 and 1
  # 1 array per input, containing the weight that input will receive in each neuron

  # Create layer 2 (a single neuron with 4 inputs)
  layer2 = NeuronLayer(1, 4)
  # 4 arrays of 1 number
  # 1 array per input, containing the weight that input will have in the (single) neuron

  # Combine the layers to create a neural network
  neural_network = NeuralNetwork(layer1, layer2)

  print ("Stage 1) Random starting synaptic weights: ")
  neural_network.print_weights()

  # The training set. 
  training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
  training_set_outputs = array([[0, 1, 1, 1, 0, 0]]).T

  # Train the neural network using the training set.
  neural_network.train(training_set_inputs, training_set_outputs, 200000)

  print ("Stage 2) New synaptic weights after training: ")
  neural_network.print_weights()

  # Test the neural network with a new situation.
  print ("Stage 3) Considering a new situation [1, 0, 0] -> ?: ")
  hidden_state, output = neural_network.think(array([1, 0, 0]))
  print (output)
