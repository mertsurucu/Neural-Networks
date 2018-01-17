import numpy as np
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

train_data = np.load('train-data.npy')
train_label = np.load('train-label.npy')

validation_data = np.load('validation-data.npy')
validation_label = np.load('validation-label.npy')



train_data=train_data/255
validation_data=validation_data/255
test_data = np.load('test-data.npy')



train_data = np.hstack((train_data,np.ones((len(train_data),1))))
validation_data = np.hstack((validation_data,np.ones((len(validation_data),1))))
LEARNING_RATE=0.005
weight=np.array(np.random.rand(10,785)/10000)
BATCH_SIZE=128
TRAIN_SIZE=50000
EPOCH = 200


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
def sigmoid_derivative(x):
        return x * (1 - x)
def relu(x):
    return x if x>0 else 0



train_label_value=np.zeros((50000,10))
for i in range(50000):#makes the label as a numpy array like[0,1,...0,0]
    train_label_value[i,train_label[i]]=1

train_label=train_label_value
# The neural network thinks.
def learn(inputs):
    return sigmoid(inputs.dot(weight.transpose()))

def gradient(batched_data, batched_label):
        for j in range(len(batched_data)):
            output = learn(batched_data[j])
            error = output - batched_label[j]
            for k in range(len(weight)):
                weight[k] -= LEARNING_RATE*(batched_data[j]*error[k])


i = 0
for j in range(EPOCH):
    print(j, ". epoch")
    while(i<TRAIN_SIZE):
        batched_data = train_data[i:i+BATCH_SIZE]
        batched_label = train_label[i:i+BATCH_SIZE]
        i = i+BATCH_SIZE
        gradient(batched_data, batched_label)
        print(i,"/",TRAIN_SIZE)



print(weight)

def accuracy(num_matches,num_test):
    return (100*num_matches)/num_test

def evalaute(data,label):
    num_matches = 0
    for i in range(len(validation_data)):
        pred = learn(validation_data[i])
        if validation_label[i]==np.where(pred==max(pred))[0][0]:
            num_matches+=1
    return num_matches/len(data)

stop = timeit.default_timer()
print(evalaute(validation_data, validation_label))
print(stop - start,"sn")


for i in range(10):
    img = weight[i][0:784]
    img = np.reshape(img,(28,28))
    plt.imshow(img, cmap='gray')
    plt.show()

