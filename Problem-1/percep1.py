import numpy as np
#from numpy import genfromtxt
import csv

# Class containing parameters used in perceptron model
class Parm:
    def __init__(self):
        # Initialize learning rate
        self.eta = 1

# Class containing weights for perceptron model
class Weight:
    # Initialize either uniformly distributed weights or zeros
    def __init__(self, n):
        # Initialize bias term 
        self.b = np.random.uniform(-1, 1)

        # Initialize weights
        self.arr = np.random.uniform(-.1, .1, n)

    # Define function to generate output of perceptron
    def percep_out(self, feature):
        return np.sign(np.dot(feature, self.arr) + self.b)

    # Define error function for perceptron
    def percep_err(self, target, output):
        return target - output

    # Define function to update weights based on error
    def update_w (self, feature, err, eta):
        # Update value of bias term
        self.b += eta * err

        # Update weight values
        self.arr = np.add(self.arr, np.dot(eta * err, feature))

    # Define function for training perceptron on data set
    def percep_train(self, feature, target, eta, n):
        # Initialize number of errors
        num_err = 0

        # Perform forward propagation for each data point
        for i in range(n):
            # Update final output value
            out = self.percep_out(feature[i])
            # Update error value
            err = self.percep_err(target[i], out)

            # Check if error is nonzero        
            if (err != 0):
                # Increase number of errors by 1
                num_err += 1
                # Update weights 
                self.update_w(feature[i], err, eta)
        
    # Define function for testing perceptron on data set
    def percep_test(self, feature, target, n):
        # Initialize number of errors
        num_err = 0

        # Perform forward propagation for each data point
        for i in range(n):
            # Update final output value
            out = self.percep_out(feature[i])
            # Update error value
            err = self.percep_err(target[i], out)

            # Check if error is nonzero     
            if (err != 0):
                # Increase number of errors by 1
                num_err += 1

        # Return the number of errors
        return num_err

def main():
    # Convert data file 'bank_numeric.csv' into two arrays: the first for
    # training and the second for testing.

    # The array train_feature has 2234 rows and 44 columns. 
    train_feature = np.genfromtxt('bank_feature.csv', skip_header=1, 
                                   skip_footer=2260, delimiter=';')
    train_target = np.genfromtxt('bank_target.csv', skip_header=1, 
                                   skip_footer=2260, delimiter=';')

    # The array test_data has 2233 rows and 44 columns.
    test_feature = np.genfromtxt('bank_feature.csv', skip_header=2262, delimiter=';')
    test_target = np.genfromtxt('bank_target.csv', skip_header=2262, delimiter=';')

    # Initialize weights for perceptron
    w = Weight(44)

    # Print initial weights for perceptron
    print("\n ------------------------ Initial Weights -------------------------")
    print("  bias = ", w.b, "\n")
    print("  w = ", w.arr)

    # Initialize parameters for perceptron
    p = Parm()
    print("\n --------------------------- Parameters ---------------------------")
    print("  Learning rate = ", p.eta)


    print("\n ------------------------- Initial Errors -------------------------")
    init_err = w.percep_test(test_feature, test_target, 2232)
    print("  Percent of misclassified features: %3.4lg%%" % (100 * init_err/2232))

    # Train over 100 epochs
    for i in range(100):
        w.percep_train(train_feature, train_target, p.eta, 2233)

    print("\n -------------------------- Final Weights -------------------------")
    print("  bias = ", w.b, "\n")
    print("  w = ", w.arr)

    print("\n -------------------------- Final Errors --------------------------")
    fin_err = w.percep_test(test_feature, test_target, 2232)
    print("  Percent of misclassified features: %3.4lg%%" % (100 * fin_err/2232))

# Run main program
main()
