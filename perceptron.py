# %%
import numpy as np

# Load train and test data
def readData(data):
    """
    Reads data and converts to numpy arrays 
    """
    dat = np.array([i.split(',') for i in open(data,"r").read().splitlines()])
    dat = dat[1:].astype(float)
    return dat

data = readData("iris.csv")
np.random.seed(3)
np.random.shuffle(data)

train_data = np.array(data[:100])
test_data = np.array(data[100:])

#%%
# Perceptron Model for Binary Classification
class Perceptron():
    """
    A perceptron model

    Class variable: -
    epochs - number of iterations for training, int.
    u - learning rate, float.
    W - weight vector, numpy array.
    num_features - number of input features, int.

    Methods: -
    actScore, actFunction, updateRule, and training
    """

    def __init__(self, epochs, num_features, learning_rate):
        """
        initialise perceptron model
        """
        self.epochs = epochs
        self.u = learning_rate
        self.W = np.zeros(num_features+1)

    def actScore(self, X):
        """
        Compute activation score: [bias + Weights.T * inputs]
        """
        return self.W[0] + np.dot(self.W[1:], X)
    
    def actFunction(self, a, threshold=0):
        """
        Converts activation score in to 1 or -1 
        """
        return 1 if a>threshold else -1
    
    def updateRule(self, X, y):
        """
        Updates weights and bias
        reg --> True/False, to use L2 regularisation
        k --> float, lambda value for L2 regularisation
        """
        if self.reg==False:
            self.W[1:] = self.W[1:] + (y*X*self.u) #Update weights
        else:
            #Update weights with L2 regularisation
            self.W[1:] = ((1-(2*self.k*self.u)) * self.W[1:]) + (y*X*self.u)

        self.W[0] = self.W[0] + self.u*y #Update Bias

    def training(self, train, reg=False, k=1):
        """
        Training the perceptron model
        """
        self.reg = reg
        self.k = k
        for i in range(self.epochs):
            for j in train:
                X = j[0:4] #Features
                y = j[-1] #Target
                a = self.actScore(X) #Computing activation score
                if y*a <= 0: #Checking for Misclassification
                    self.updateRule(X, y) #Update weight and bias

def prepData(data, classMap):
    """
    Prepare train/test data
    classMap --> Map dictionary to prepare data
    """
    #Select rows for given classes
    dat = np.array([i for i in data if i[-1] in classMap.keys()])
    #Convert class values to given targets
    dat[:,4] = [classMap[i] for i in dat[:,4]]
    return dat

def accuracy(data, model):
    """
    Computes accuracy of predictions and target labels
    """
    pred = [model.actFunction(model.actScore(i[0:4])) for i in data] 
    #computing accuracy
    acc = sum(1 for x,y in zip(data[:,4], pred) if x == y) / float(len(data)) 
    return round(acc*100, 2)

#%%

def train_test(epochs, learning_rate, reg, train_data, test_data, classMap):
    """
    Runs training and testing
    """
    ft_size = train_data.shape[1]-1 #Number of input features
    train = prepData(train_data, classMap)
    test = prepData(test_data, classMap)
    model = Perceptron(epochs=epochs, num_features=ft_size, learning_rate=learning_rate)
    model.training(train=train, reg=reg)
    print("Train Accuracy is", accuracy(train, model), "%")
    print("Test Accuracy is", accuracy(test, model), "%\n")

#### Class 0 and 1 ####
print("Class 0 and 1")
train_test(epochs=20, learning_rate=1, reg=False, train_data=train_data, 
                                    test_data=test_data, classMap={0:1, 1:-1})

#### Class 1 and 2 ####
print("Class 1 and 2")
train_test(epochs=20, learning_rate=1, reg=False, train_data=train_data, 
                                    test_data=test_data, classMap={1:1, 2:-1})

#### Class 0 and 2 ####
print("Class 0 and 2")
train_test(epochs=20, learning_rate=1, reg=False, train_data=train_data, 
                                    test_data=test_data, classMap={0:1, 2:-1})

# %%
#### One Vs Rest: Multiclass classifier model ####

# Defining Methods for multi classifier
def OneVSrest(train, ft_size, epochs=20, reg=False, k=1):
    """
    Multi class classifier implementation using One Vs Rest method
    reg--> regularisation flag
    k--> lambda
    """
    # Model initialisation and training
    model1 = Perceptron(epochs=epochs, num_features=ft_size, learning_rate=1)
    model1.training(train=train[0], reg=reg, k=k)

    model2 = Perceptron(epochs=epochs, num_features=ft_size, learning_rate=1)
    model2.training(train=train[1], reg=reg, k=k)

    model3 = Perceptron(epochs=epochs, num_features=ft_size, learning_rate=1)
    model3.training(train=train[2], reg=reg, k=k)

    return model1, model2, model3

def regAccuracy(data, model1, model2, model3):
    """
    Computes accuracy for multiclassifier model
    """
    preds = [np.argmax([model1.actScore(i[0:4]), model2.actScore(i[0:4]), 
                                    model3.actScore(i[0:4])]) for i in data]
    acc = sum(1 for x,y in zip(data[:,4], preds) if x == y) / float(len(data))
    return round(acc*100,2)

#Prepare training data
train1 = prepData(train_data, classMap={0:1, 1:-1, 2:-1})
train2 = prepData(train_data, classMap={0:-1, 1:1, 2:-1})
train3 = prepData(train_data, classMap={0:-1, 1:-1, 2:1})

ft_size = train_data.shape[1]-1 #Number of input features

# Initialise and train multiclassifier
model1_23, model2_13, model3_12 = OneVSrest(train=[train1, train2, train3], 
                                    ft_size=ft_size, epochs=20, reg=False, k=0)

# Train and Test accuracy
print("\nTrain Accuracy for Multi class classifier is", 
                 regAccuracy(train_data, model1_23, model2_13, model3_12), "%")
print("Test Accuracy for Multi class classifier is", 
                  regAccuracy(test_data, model1_23, model2_13, model3_12), "%")

#%%
# Regularisation
print("Regularisation:-")
for k in [0.01, 0.1, 1.0, 10.0, 100.0]:
    # Initialise and train model with lambda(k) values
    modelA, modelB, modelC = OneVSrest(train=[train1, train2, train3], 
                                     ft_size=ft_size, epochs=20, reg=True, k=k)
    print("\nTrain Accuracy for Multi class classifier with k =",k,"is", 
                        regAccuracy(train_data, modelA, modelB, modelC), "%")
    print("Test Accuracy for Multi class classifier with k =",k,"is", 
                        regAccuracy(test_data, modelA, modelB, modelC), "%")
# %%
