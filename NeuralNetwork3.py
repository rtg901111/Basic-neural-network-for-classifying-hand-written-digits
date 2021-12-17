import numpy as np
import time
import sys

def read_input():
    
    files = sys.argv
    
    train_image = files[1]
    train_label = files[2]
    test_image = files[3]
    
    train_image = np.genfromtxt(train_image, delimiter = ',')
    
    print('Train_image_read')
    train_label = np.genfromtxt(train_label, delimiter = ',')
   
    print('Train_label_read')
    test_image = np.genfromtxt(test_image, delimiter = ',')
    print('Test_image_read')

    '''
    train_image = np.genfromtxt('train_image.csv', delimiter = ',')
    
    print('Train_image_read')
    train_label = np.genfromtxt('train_label.csv', delimiter = ',')
   
    print('Train_label_read')
    test_image = np.genfromtxt('test_image.csv', delimiter = ',')
    print('Test_image_read')
 
    #print('Test_image_read')

    #temp = np.array_split(test_image, 10)
    '''
    
    return train_image, train_label, test_image

def sigmoid(arr):
    arr = np.clip(arr, -50, 50)

    after = 1 / (1 + np.exp(-arr))
    return after
 
def sigmoid_deriv(arr):
    arr = np.clip(arr, -50, 50)

    temp = np.exp(-arr)
    after = temp / ((temp + 1) ** 2)
    return after

def softmax(arr):
    exponent = np.exp(arr - arr.max())
    after = exponent / np.sum(exponent)
    return after 

def softmax_deriv(arr):
    exponent = np.exp(arr - arr.max())
    temp = np.sum(exponent, axis = 0)
    after = exponent / temp * (1 - exponent / temp)
    return after

def logistic_loss(predicted_output, true_output):
    return -(true_output * np.log(predicted_output) + (1-true_output) * np.log(1 - predicted_output))

class neuralnetwork:
     
    def __init__(self, neurons):
        self.neurons = neurons

        self.learning_rate = 0.05 #0.01 about 75% acc, 0.02 = 79% acc, 0.05 = 81% acc
        self.weight = {}
        self.bias = {}

        self.after_act = {}
        self.before_act = {}

        for i in range(3):
            self.weight[i + 1] = np.random.randn(self.neurons[i + 1], self.neurons[i])
            self.bias[i + 1] = np.zeros((self.neurons[i + 1], 1))

    def feed_forward(self, arr):

        self.after_act[0] = arr
        #print(np.isfinite(self.weight[1]).all())
                                    #   196 x 784     784, 1
        #print(self.weight[1].shape, self.after_act[0].shape, "adlfasdfasdfsa")
                 
        
        #self.before_act[1] = self.weight[1].dot(self.after_act[0]) + self.bias[1]

        #print(self.before_act[1])
        #print(self.before_act[1].shape)
        #self.after_act[1] = sigmoid(self.before_act[1])
        #print(self.after_act[1].shape)
        #print(self.after_act[1]) 

        #self.before_act[2] = self.weight[2].dot(self.after_act[1]) + self.bias[2]
        #self.after_act[2] = sigmoid(self.before_act[2])

        for i in range(2):
            self.before_act[i+1] = self.weight[i+1].dot(self.after_act[i]) + self.bias[i+1]
            self.after_act[i+1] = sigmoid(self.before_act[i+1])
 

        #print(self.after_act[2])

        #self.before_act[3] = np.dot(self.weight[3], self.after_act[2]) + self.bias[3] 
        self.before_act[3] = self.weight[3].dot(self.after_act[2]) + self.bias[3] 
        self.after_act[3] = softmax(self.before_act[3])
        #print(self.after_act[3])

        return self.after_act[3]

    def back_prop(self, loss):
        
        weight_loss = loss / 10 * softmax_deriv(self.before_act[3])
        self.weight[3] -= np.outer(weight_loss, self.after_act[2]) * self.learning_rate
        self.bias[3] -= np.sum(weight_loss, axis = 1, keepdims=True) * self.learning_rate

        #print(weight_loss)

        #print(self.weight[3])

        #print(weight_loss.shape, self.weight[3].T.shape, "weight_loss.shape")

        #weight_loss = np.dot(self.weight[3].T, weight_loss) * sigmoid_deriv(self.before_act[2])
        weight_loss = self.weight[3].T.dot(weight_loss) * sigmoid_deriv(self.before_act[2])
        self.weight[2] -= np.outer(weight_loss, self.after_act[1]) * self.learning_rate
        self.bias[2] -= np.sum(weight_loss, axis = 1, keepdims = True) * self.learning_rate

        #print(self.weight[2])

        #weight_loss = np.dot(self.weight[2].T, weight_loss) * sigmoid_deriv(self.before_act[1])
        weight_loss = self.weight[2].T.dot(weight_loss) * sigmoid_deriv(self.before_act[1])
        self.weight[1] -= np.outer(weight_loss, self.after_act[0]) * self.learning_rate
        self.bias[1] -= np.sum(weight_loss, axis = 1, keepdims = True) * self.learning_rate

        #print(self.weight[1])

        #dsdsdsdfaf

def predict(network, test_image):
    predictions = []

    for image in test_image:
        image = image.reshape(image.shape[0], 1)
        pred = network.feed_forward(image)

        pred = np.argmax(pred)

        predictions.append(int(pred))
    
    predictions = np.array(predictions)
    predictions = predictions.reshape((len(predictions), 1))
    
    np.savetxt('test_predictions.csv', predictions, delimiter = ',')


#def fit(epochs, train_image, train_label, test_image):
def fit(epochs, train_image, train_label, test_image):
    accuracy_list = []
    
    network = neuralnetwork([784, 532, 261, 10]) #784, 196, 64, 10
    for epoch in range(epochs):

        cnt = 0

        batches_image = np.array_split(train_image, 200) #100 #60
        #batches_image = np.split(train_image, np.arange(128, len(train_image), 128))
        batches_label = np.array_split(train_label, 200) #100 #60
        #batches_label = np.split(train_label, np.arange(128, len(train_label), 128))

        #For each 1000 rows
        for image_list, label_list in zip(batches_image, batches_label):
            #if cnt % 10 == 0:
            #    print('batch ' + str(cnt) + ' started')
            #For each row
            for image, label in zip(image_list, label_list):
                image = image.reshape(image.shape[0], 1)
                after_act_3 = network.feed_forward(image)
                
                label_1d = np.array([0] * 10)
                label_1d[int(label)] = 1
                label_1d = label_1d.reshape(10, 1)
                #print(label_1d.shape, after_act_3.shape, " label_1d, aftera_act3 shape")
                #print(label_1d)

                #loss = logistic_loss(after_act_3, label_1d)
                loss = 2 * (after_act_3 - label_1d)
                #print(loss.shape, "loss_shape")

                network.back_prop(loss)

            cnt += 1

        
        #accuracy_list.append(accuracy_rate)
        #print(accuracy_rate)
    
    #print(accuracy_list)
    predict(network, test_image)
    

if __name__ == '__main__':
    start_time = time.time()

    train_image, train_label, test_image = read_input()
    epochs = 80 #70 #50 is the best with 0.80 accuracy with learning_rate = 0.05

    fit(epochs, train_image, train_label, test_image)
     
    print('Took', str(time.time() - start_time), 'seconds')
     
 
 #split = 200, epoch = 100 -> 83.6% better than 100/100
 #epochs 100 the best  