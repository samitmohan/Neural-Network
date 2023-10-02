Pytorch : Deep Learning Framework which accelerates the model training process
Built by facebook,

Tensors : Multi dimensional Arrays (5D+ objects)

Building a neural network using pytorch.

Deep Learning represents prediction (generalise rules of operation : correc
behaviour)

Philosophy = Rules do not matter, data does.

Uses ANN (Input -> Hidden -> Output) using backpropagation / gradient descent

Thousands of samples are required for training -> Train -> Predict

Can use pre-trained model and then fine tune it as per our liking -> make
predictions -> Pytorch.

Built around tensor class, has high performance on GPUs with Cuda, built in back
propagation (autograd), neural network building blocks. (Python + C++)

Adam Optimizer : algo in DL, helps adjust parameters of neural network in real
time to improve accuracy + speed. Adaptive Moment Estimation. (Helps NN learn
faster and converge quickly towards optimal set of parameters that minimize the
loss function)

One of the main advantages of Adam is its ability to handle noisy and sparse
datasets, which are common in real-world applications

# Gradient Descent

Gradient descent is an optimization algorithm commonly used in machine learning
and deep learning to minimize a cost or loss function. Its primary purpose is to
find the optimal parameters (weights and biases) for a model that minimizes the
error between predicted and actual values.

Steps -:
Calculate Gradient (derivative) of cost function with respect to each parameter.
update_parameter -= learning_rate \* gradient(update_parameter)
learning rate = hyper parameter that controls size of each step.
Repeat these two steps until cost function converges to minimum value

Stochastic Gradient Descent (SGD): Instead of using the entire dataset to
compute the gradient at each iteration, it uses a single random data point (or a
small batch of data) to estimate the gradient

Hence Adam is an extension to GD algorithm

## NN Layers

Input Layer : accepts raw data
Conv2D Layer : process grid like data (images) -> extract features from input
data (filters)
Activation Layer (ReLU) : applies simple threshold function (patterns in data)
Pooling Layer : Downsample the feautre maps produced by convolutional layers.
{max_pooling selects max value from group of values in local region on input}
Flattern Layer : Convert multi-dimensional output of previous layer -> 1D Vector

Fully Connected Layer (Linear) : connects every neuron from previous layer to
every neuron in current layer. (final)

Other layers
Dropout : prevents overfitting
Batch Normalization : normalize input to a layer in order to stabilize
Softmax : Computes probabilities of each class : sum upto 1.

## Cross Entropy

"cross-entropy" is a commonly used loss function in machine learning and deep
learning, particularly for classification tasks. It measures the dissimilarity
or "distance" between the predicted probabilities (or scores) and the true
labels of the data

#### Learnings from Neural Network + Deep Learning chapter from Data Science for Python (basics)

Neural Network -:

- Perceptrons
