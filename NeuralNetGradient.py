from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):

        #seed random number
        random.seed(9)

        # Define a neuron, with 4 input connections and 1 output connection
        # Assign random weights to a 4 x 1 matrix, with values scaled in the range -1 to 1 and mean 0
        self.weights = 2 * random.random((4, 1)) - 1

    # The Sigmoid function, logsig or tanh
    # Pass the weighted sum of the inputs through this function to normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Gradient of the Sigmoid curve.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Adjusting the weights each iteration 
    def train(self, training_inputs, training_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through the neural network
            output = self.predict(training_inputs)

            # Calculate the error (The difference between the desired output and the predicted output).
            error = training_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid
            # Inputs which are zero, do not cause changes to the weights.
            adjustment = dot(training_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.weights += adjustment

    # Network Output Prediction
    def predict(self, inputs):
        # Pass the inputs through the neural network
        return self.__sigmoid(dot(inputs, self.weights))



if __name__=="__main__":
    neural_network=NeuralNetwork()

    print("Initializing random weights: ")
    print(neural_network.weights)

    #defining training inputs and outputs
    training_inputs=array([[0,0,1,0],[1,1,1,1],[1,0,1,1],[0,1,1,1]])
    training_outputs=array([[0,1,1,0]]).T

    #train 5000 iterations 
    neural_network.train(training_inputs,training_outputs,5000)

    print("Weights after Network training: ")
    print(neural_network.weights)

    # Test the neural network
    print("Prediction [0, 1, 0, 1] -> : ")
    print(neural_network.predict(array([0, 1, 0, 1])))
    
