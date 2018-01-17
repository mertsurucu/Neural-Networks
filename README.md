# Neural-Networks
Parameterized Neural Network Implementation
## PART II: Classification of Fashion Products

## using Neural Networks

The MNIST dataset is (seemingly) the most abused dataset for beginning with image
classification. It contains **28x28 grayscale images of handwritten digits** , each with an
associated label indicating which number is written (an integer between 0 and 9).

The Fashion MNIST training set contains **55,000** examples, and the test set contains **10,**

examples. Each example is a **28x28 grayscale image** (just like the images in the original

MNIST), associated with a **label from 10 classes** (t-shirts, trousers, pullovers, dresses, coats,

sandals, shirts, sneakers, bags, and ankle boots). Each training and test example is assigned

to one of the following labels:

```
Label Description
```
```
0 T-shirt/top
```
```
1 Trouser
```
```
2 Pullover
```
```
3 Dress
```
```
4 Coat
```
```
5 Sandal
```
```
6 Shirt
```
```
7 Sneaker
```
```
8 Bag
```
```
9 Ankle boot
```

### 1 - Single-Layer Neural Network

```
The input layer will consist
of 28 .28=784 neurons, corresponding
with the pixels in an instance image
so, every image is flattened in
advance, getting a 1 x 784 vector
representation of the originally
2 Dimensional image.
```
I trained the network feeding by given training set as 784 dimensional gray-level image
values. It is important to normalize image values (0 -255) to between 0 and 1. So I divided
whole value by 255 to use numbers between 0 to 1.

### We can express this network mathematically as: Ɵi = wijxj + bi

```
In this example we have 3 different
inputs and simultaneously 3 different
weights.
First, I initialized random weights
between 0 to 1. Randomly selected
weights between 0 to 1 gives me high
numbers for sigmoid function. So
that, I divide them by 1000 for putting
it to sigmoid function.
I wrote a function to compute loss
function.
```
After that I updated network parameters weight and bias to minimize the loss function using
gradient descent algorithm.
Then I wrote the derivative of the loss
function with respect to the parameters. I
also implemented numerical
approximation of gradients. To minimize
the cost function, I needed to use mini-
batch gradient descent.
I have chosen different learning rates
0.005, 0.01, 0.015 but I got the best result
when the learning rate is small (0.005).


One epoch is one forward pass and one backward pass of all the training examples.
Batch size is the number of training examples in one forward/backward pass. The higher the
batch size, the more memory space you'll need.
```
```
I chose the batch size 128 because it executes faster, and I got better predictions, also I
chose epoch size 150 to get better result. I tried different epochs to prevent overfit and
underfit.
For each epoch, I carried out training sequentially on minibatches. Finally, I saved the
parameters and calculated the model’s accuracies on the training and test.
I visualized the weights for each set of parameters that connect to O0; O1; ; ; ; ;O9.
As a result, the weight of each class represents the visual represented by that class. For
example, we can see that all the shoes representation in the shoes weights.
```
### 2 - Multi-Layer Neural Network

```
(In this part I couldn’t implement the multi-layer neural network so, instead of implementing
it; I used Keras)
```
```
In contrast, the output layer will
have just 10 neurons one per class
because at the output layer, the
net will compute the probability of
a sample belongs to every class,
from which we will take the
neuron associated to the greatest
value as the predicted class.
```
### Forward propagation

```
Forward propagation is how our neural
network will make a prediction given our input
image. To do this, we will carry out a series of
```
#### steps. function which takes in the input

```
image X (a NumPy array) and
our parameters dictionary (which contains the
weights and biases for each layer), and returns
the output from the last linear unit.
```

### Computing cost

We can create a function to compute the cost based on the output from the last linear layer

and the actual target classes. The cost is just a measure of the difference between the target

### class predicted by our neural network and the actual target class.

```
This cost will be used during backpropagation to update the weights. The greater the cost,
the greater the magnitude of each parameter update.
```
### Backpropagation

```
Backprop is an essential step for our neural network, because it decides how much to
update our parameters (weights and biases) based on the cost from the previous
computation.
learning_rate (learning rate of the optimisation)
num_epochs (number of epochs of the optimisation loop)
batch_size (size of a batch)
Architecture for neural networks, consisting of a variable number of layers where the
information from one layer to the next one is performed by a linear operation y=WT⋅x+b,
which is the dot product of the corresponding weight matrix by the input at that layer plus
the bias term. Tanh functions, defined as:
```
```
To have an intuition on how many hidden layers yield the best score, I ran a series of trials
where the only change among the nets were the number of hidden layers, but the number
of epochs, learning rate and number of neurons in each hidden layer is the same. Iterations
training, which is around 30 epochs considering that batch size=128, the accuracies obtained
over the test were set.
```
```
The input and output layer must remain the same due to the task. The hidden layer should
have many neurons such that it does not jam but neither should be too large.
The experiments I have run cover a wide range of sizes: from 600 to 10 neurons: 600 , 300 ,
100 and 10. The learning rate and the training time remain the same than in the previous
experiments.
```
```
What I observe is that too few neurons yield a poor result. I can check that the nets we were
working with in the previous experiment could likely perform much better if more neurons
had been added to the models. I have noticed that an intermediate number of neurons in
the hidden layer is the best choice. Because an intermediate number of neurons allows the
net to properly learn the most remarkable features of the dataset, making a reasonable
generalisation of it.
