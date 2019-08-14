import numpy as np
import csv

# Define class to contain parameters for neural network
class Parm:
    # Initialize parameters
    def __init__(self, b_size, data_rows, data_cols, uniform):
        # Initialize learning rate
        self.eta = 1

        # Initialize data row and column sizes
        self.rows = data_rows
        self.cols = data_cols

        # Initialize batch size
        self.batch_size = b_size

        # Initialize batch
        self.batch = np.random.randint(0, self.rows, self.batch_size)

        # Initialize either uniformly distributed weights or zeros
        if (uniform):
            d = 4 * np.sqrt(6/(1 + self.cols))
            # Initialize bias term 
            self.b = np.random.uniform(-d, d)

            # Initialize weights
            self.w = np.random.uniform(-d, d, self.cols)
        else:
            # Initialize bias term to zero
            self.b = 0

            # Initialize weights to zeros
            self.w = np.zeros(self.cols)

        # Initialize output value
        self.o = 0

        # Initialize error value
        self.Err = 0

        # Initialize delta W
        self.del_w = np.zeros(self.cols)

        # Initialize delta b
        self.del_b = 0

    # Routine to set all del terms equal to zero
    def reset_del(self):
        self.del_w = np.zeros(self.cols)
        self.b = 0

    # Modify values in batch
    def reset_batch(self):
        self.batch = np.random.randint(0, self.rows, self.batch_size)

    # Define function to generate output of neural network
    def update_o(self, feature):
        # Compute value at which to evaluate tanh function
        temp = np.dot(feature, self.w) + self.b
        self.o = np.tanh(temp)
#        return self.o #sigm(temp)

    # Define error function for neural network
    def update_Err(self, target):
        self.Err = target - self.o
        return self.Err #target - output

    # Define function to compute error over set of specified indices
    def tot_err(self, feature, target, indices):
        # Initialize temp
        temp = 0

        # Compute error term for each index in set
        for i in np.nditer(indices):
            out = self.update_o(feature[i])
            temp += 0.5 * (target[i] - self.o)**2

        # Return error
        return temp

    # Define function for forward propagation
    def forward(self, f_row, feature):
        self.update_o(feature[f_row])

    # Define function to perform backpropagation
    def backward(self, f_row, feature, target):
        # Compute Error terms for current data point
        self.update_Err(target[f_row])
        # Compute gradient values for current data point
        temp = self.eta * self.Err * (1 - self.o**2) / self.batch_size
        # Update del_w and del_b
        self.del_b += temp

        # Update del_w
        for j in range(self.cols):
            self.del_w[j] += temp * feature[j]

    # Define function to update weights based on error
    def update_w (self):
        # Update bias term
        self.b += self.eta * self.del_b

        # Update weights
        for j in range(self.cols):
            self.w[j] += self.eta * self.del_w[j]

    # Routine to run batch gradient descent
    def batch_grad_descent(self, feature, target):
        for j in self.batch:
            # Perform forward propagation
            self.forward(j, feature)

            # Perform backward propagation
            self.backward(j, feature[j], target)

        # Update weights
        self.update_w()

    # Define function for testing neural network on data set
    def nnet_test(self, feature, target):
        # Initialize number of errors
        num_err = 0

        # Perform test for all values in given data set
        for i in range(self.rows):
            # Compute output value for given data point
            self.update_o(feature[i])

            # Check if target and ourput are not equal
            if (target[i] != np.sign(self.o)):
                # Update number of errors
                num_err += 1

        print(" Number of testing errors = ", num_err)
        return num_err

# Main function
def main():

    # Convert data file 'bank_numeric.csv' into two arrays: the first for
    # training and the second for testing.

    # The array train_feature has 2234 rows and 44 columns. 
    train_feature = np.genfromtxt('bank_feature.csv', skip_header=1, 
                                   skip_footer=2260, delimiter=';')
    train_target = np.genfromtxt('bank_target.csv', skip_header=1, 
                                   skip_footer=2260, delimiter=';')
    train = np.genfromtxt('bank_numeric.csv', skip_header=1, 
                           skip_footer=2260, delimiter=';')

    # The array test_data has 2233 rows and 44 columns.
    test_feature = np.genfromtxt('bank_feature.csv', skip_header=2262, delimiter=';')
    test_target = np.genfromtxt('bank_target.csv', skip_header=2262, delimiter=';')

    # Initialize weights and parameters for neural network
    w = Parm(15, 2233, 44, True)

    # Initialize rows to use in computation of total error
    a = range(0, w.rows - 1)


    # Print initial parameters for neural network
    print("\n ------------------------ Initial Weights ------------------------")
    print("  Initial bias = ", w.b)
    print("  Initial w = \n", w.w)

    print("\n ----------------------- Initial Parameters -----------------------")
    print("  Learning rate = ", w.eta)
    print("  Batch size = ", w.batch_size)

    print("\n ------------------------- Initial Errors -------------------------")
    print("  Error function for training values at start: ",
          w.tot_err(train_feature,train_target,a))
    init_err = w.nnet_test(train_feature, train_target)
    print("  Percent of misclassified training features: "
          " %3.4lg%%" %(100 * init_err/2233))
    print("  Error function for testing values at start: ",
          w.tot_err(test_feature,test_target,a))
    init_err = w.nnet_test(test_feature, test_target)
    print("  Percent of misclassified testing features: "
          " %3.4lg%%" % (100 * init_err/2233))

    for j in range(100):
        # Reset the values in the batch
        w.reset_batch()

        # Reset the values in del
        w.reset_del()

        # Perform the batch stochastic gradient descent
        w.batch_grad_descent(train_feature, train_target)

    print("\n ------------------------- Final Weights -------------------------")
    print("  Final bias = ", w.b)
    print("  Final w = \n", w.w)

    print("\n ------------------------- Final Errors --------------------------")
    print("  Error function for training values at finish: ",
          w.tot_err(train_feature,train_target,a))
    fin_err = w.nnet_test(train_feature, train_target)
    print("  Percent of misclassified training features: "
          " %3.4lg%%\n" % (100 * fin_err/2233))
    print("  Error function for testing values at finish: ",
          w.tot_err(test_feature,test_target,a))
    fin_err = w.nnet_test(test_feature, test_target)
    print("  Percent of misclassified testing features: "
          " %3.4lg%%" % (100 * fin_err/2233))


# Run main program
main()
