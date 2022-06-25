import numpy as np

"""
A simple ML library created from scratch using only numpy with one class per layer. 
This is for demonstration but feel free to use for anything.
Basic knowledge of linear algebra and multi-layer feed-forward neural network design is required to understand.
We start by making our base input layer.
"""
class Layer:
    def __init__(self):
        # These attributes are so other layers don't have to declare them.
        self.input = None
        self.output = None

    def forward(self, input):
        # Takes in input gives output
        # TODO: return output
        pass

    def backward(self, output_grad, learning_rate):
        # Takes in derivative of error w.r.t output and updates the trainable parameters
        # also returns the derivative of the error w.r.t input
        # TODO: return output
        pass

"""
Now we make the dense layer. This inherits from the base layer class above. 
In the dense layer each input neurons connects to each output neuron.
Each connection represents a weight and every output value is calculated as the sum of all the inputs and their respective weights connecting them to the output.
A bias is added to this output at the end.
The easiest way to make this process in our code is with matrix multiplication. The weight matrix is dimension j*i and the bias is a column vector.
"""
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Constructor initialises weights and biases randomly
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        # Computes Y = W*X + b using dot product
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Here, we implement backward propagation of the dense layer.
        It calculates the gradient of the error function with respect to the weights.
        The calculation of the gradient proceeds backwards through the network, with the gradient of the final layer of weights 
        being calculated first and the gradient of the first layer of weights being calculated last.
        It is possible because the gradient input of one layer is the gradient output of the previous layer, 
        therefore we can use chain rule to get efficient computation of the gradient at each layer. 
        """
        weights_gradient = np.dot(self, output_gradient, self.input.T) # dE/dW = dE/dY * X.T, also notice: dE/dB = dE/dY
        self.weights -= learning_rate * weights_gradient # This and the next line implement gradient descent
        self.bias -= learning_rate * output_gradient 
        return np.dot(self.weights.T, output_gradient) # dE/dX = W.T * dE/dY 

"""
Next, we implement the activation layer because there was no activation function in the previous layer.
Notice that it was simply Y = W*X + b and not Y = f(W*X) + b. This keeps the code simple and readable. 
The activation layer takes in some neurons and simply passes them through an activation function. 
Therefore output layer and input have the same dimensions. The forward propagation is simply Y = f(X).
"""
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation(self.input)) # Element-wise multiplication of the two vectors dE/dY and f'(X)

"""
Let's implement one non-linear activation function: the hyperbolic tangent function.
All we need is the function f(x) = tanh(x) and its derivative f'(x) = 1 - tanh(x)^2.
"""
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x) # f(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2 # f'(x)
        super().__init__(tanh, tanh_prime) # Passes in the superconstructor

"""
Finally, we must implement a loss function. We will use mean squared error for this. 
This function calculates how far the prediction is from the true value. 
We know dE/dY was given in back propagation from the input dE/dX of the next layer.
Therefore, dE/dY at the final layer is simply the output of the whole network. We can simply use this output to calculate the error.
"""
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)