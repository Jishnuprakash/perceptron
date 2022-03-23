- Requirements: -
  python 3, numpy (pip install numpy)

- Folder Structure
Please make sure, the files perceprtron.py and iris.csv are in the same location/folder. 

- How to run the code?
After installing python and numpy, it is very easy to run the perceptron.py file. 
1) Open the code in any Python IDE and click on run
or
2) Open command prompt from project folder and run "python /perceptron.py"

- Code in Detail
1) Importing libraries
Importing numpy library for array operations and numpy random module for shuffling the data
Set random seed to constant, to get same accuracy everytime

2) Reading the data
readData Method is used to read iris.csv file and convert to numpy array.

3) Perceptron model
Class Perceptron represents a perceptron model. it has following methods, actScore, 
actFunction, updateRule, and training.

4) Prepare data
prepData method is used for preprocessing the train/test data. 
uses classMap dictionary to map class values to 1 and -1.

5) accuracy
accuracy method is used to compute accuray from predictions and target variables.
a flag parameter is used to switch between model/function predictions

6) Binary CLassification (0&1, 1&2, 0&2)
Perceptron class is utilised to initialise and train 3 models to seperate data points
that belongs to class 0&1, class 1&2, and class 0&2.
Train and test accuracy for each model is printed to the console.

7) train_test
Runs training and testing and prints the accuracy

8) One VS Rest method
OneVsRest method is defined to initialise and train 3 seperate models to implement
one vs rest approach for multi-class CLassification.
Additioal parameters reg (True/False) and k (float) are used to use regularisation.

9) regAccuracy
this method is defined to compute accuracy of one vs rest method.

10) regularisation
For different values of lambda, the one vs rest multi-classifier is trained.
Train and test accuracy for each lambda value is printed to the console.


