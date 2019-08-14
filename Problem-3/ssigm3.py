import numpy as np
import pylab
import matplotlib.pyplot as plt

# Define sigmoid function
def sigm(x):
    # To prevent overflow, we use the following stable implementation
    return 1./(1. + np.exp(-x))

# Define class to contain feature and target arrays
class Data: 
    # Initialize feature and target arrays
    def __init__(self, f_col, t_col, file_name):
        # Initialize column indices of file which correspond to feature
        self.fcol = range(0, f_col)

        # Initialize column indices of file which correspond to target
        self.tcol = range(f_col, f_col + t_col)

        # Initialize array containing feature values
        self.feature = np.genfromtxt(file_name, skip_header=1, 
                                     usecols=self.fcol, delimiter=',')

        # Initialize array containing target values
        self.target = np.genfromtxt(file_name, skip_header=1, 
                                    usecols=self.tcol, delimiter=',')

# Define class to contain parameters for neural network
class Parm:
    # Initialize parameters
    def __init__(self, tolerance, eta, b_size, data_rows):
        # Initialize tolerance
        self.tol = tolerance

        # Initialize learning rate
        self.eta = eta

        # Initialize batch size
        self.batch_size = b_size

        # Initialize data row and column sizes
        self.rows = data_rows

        # Initialize current minimum error for gradient descent
        self.min_err = 1e10

        # Initialize FULL parameter to TRUE if batch_size >= 
        # If batch size is at least number of rows in data set then
        # the batch is set so that full gradient descent is used. This
        # parameter is set to True if full gradient descent is used and
        # False otherwise
        if (self.batch_size >= self.rows):
            self.FULL = True
        else:
            self.FULL = False

        # Initialize batch
        if (self.FULL):
            self.batch = range(0, self.rows)
        else:
            self.batch = np.random.randint(0, self.rows, self.batch_size)

    # Routine for updating the value of min_err
    def update_min_err(self, x):
        self.min_err = x

    # Routine for resetting the values in the batch
    def reset_batch(self):
        if not(self.FULL):
            self.batch = np.random.randint(0, self.rows, self.batch_size)

    # Routine for updating size of batch
    def update_batch_size(self, b_size):
        if not(self.FULL):
            self.batch_size = b_size

# Define class for node of recurrent neural network
class Rnn:
    # Initialize elements in class
    def __init__(self, n_layers, n_inputs, n_time, uniform):
        # Initialize number of nodes at each time step
        self.n = n_layers

        # Initialize number of inputs for each node
        self.i = n_inputs

        # Initialize total number of timesteps
        self.t = n_time

        # Initialize output errors
        self.Err = np.zeros(self.n)

        if (uniform):
            # Initialize input weights
            self.Wi = np.random.uniform(-1, 1, (self.n, self.i))

            # Initialize bias terms
            self.b = np.random.uniform(-1, 1, (self.n, self.i))

            # Initialize hidden weights
            self.Wh = np.random.uniform(-1, 1, (self.n, self.i))

        else:
            # Initialize input weights
            self.Wi = np.array([[-1.00897], [0.726748]])

            # Initialize bias terms
            self.b = np.array([[1.01231], [-0.106137]])

            # Initialize hidden weights
            self.Wh = np.array([[-0.919464], [-0.797705]])

        # Initialize final input weights (to be returned at end of program)
        self.Wi_final = np.random.uniform(-1, 1, (self.n, self.i))

        # Initialize final bias terms (to be returned at end of program)
        self.b_final = np.random.uniform(-1, 1, (self.n, self.i))

        # Initialize final hidden weights (to be returned at end of program)
        self.Wh_final = np.random.uniform(-1, 1, (self.n, self.i))

        # Initialize t by n matrix, O, of output values where the
        # entry O_{i,j} represents the output of the jth node at
        # timestep i
        self.O = np.zeros((self.t, self.n))

        # Initialize Hessian of final output with respect to Wi
        self.H_Wi = np.zeros((self.n, self.n))

        # Initialize Hessian of final output with respect to b
        self.H_b = np.zeros((self.n, self.n))

        # Initialize Hessian of final output with respect to Wh
        self.H_Wh = np.zeros((self.n, self.n))

        # Initialize delta_Wi for gradient descent
        self.del_Wi = np.zeros((self.n, self.i))
        
        # Initialize delta_b for gradient descent
        self.del_b = np.zeros((self.n, self.i))

        # Initialize delta_Wi for gradient descent
        self.del_Wh = np.zeros((self.n, self.i))

    # Routine to update the output at a given time 
    def update_O(self, input_data, timestep):
        # Check row index
        if (timestep == 0):
            # Update outputs at 0th timestep
            for j in range(0, self.n):
                self.O[0, j] = sigm(np.dot(self.Wi[j], input_data[j]) + self.b[j])

        else:
            # Update outputs for all other timesteps
            for j in range(0, self.n):
                # Compute index for input of jth node in timestep
                k = (j - 1) % self.n

                # Compute value within the network then apply sigmoid function
                temp = np.dot(self.Wh[j], self.O[timestep - 1, k]) + self.b[j]
                self.O[timestep, j] = sigm(temp)

    # Routine to update Hessians of output with respect to Wi, b, and Wh
    def update_H(self, f_row, feature):
        # ------------------ Update H_Wi ------------------ #
        # Update H_Wi[0,0]
        temp = self.O[2,0] * (1 - self.O[2,0]) * self.Wh[0]
        temp *= self.O[1,1] * (1 - self.O[1,1]) * self.Wh[1]
        temp *= self.O[0,0] * (1 - self.O[0,0]) * feature[f_row,0]
        self.H_Wi[0,0] = temp

        # H_Wh[0,1] = 0

        # H_Wh[1,0] = 0

        # Update H_Wh[1,1]
        temp = self.O[2,1] * (1 - self.O[2,1]) * self.Wh[1]
        temp *= self.O[1,0] * (1 - self.O[1,0]) * self.Wh[0]
        temp *= self.O[0,1] * (1 - self.O[0,1]) * feature[f_row,1]
        self.H_Wi[1,1] = temp

        # ------------------ Update H_b ------------------- #
        # Update H_b[0,0]
        temp = self.O[1,1] * (1 - self.O[1,1]) * self.Wh[0]
        temp *= self.O[0,0] * (1 - self.O[0,0]) * self.Wh[1]
        temp += 1
        temp *= self.O[2,0] * (1 - self.O[2,0])
        self.H_b[0,0] = temp

        # Update H_b[0,1]
        temp = self.O[2,1] * (1 - self.O[2,1]) * self.Wh[1]
        temp *= self.O[1,0] * (1 - self.O[1,0])
        self.H_b[0,1] = temp

        # Update H_b[1,0]
        temp = self.O[2,0] * (1 - self.O[2,0]) * self.Wh[0]
        temp *= self.O[1,1] * (1 - self.O[1,1])
        self.H_b[1,0] = temp

        # Update H_b[1,1]
        temp = self.O[1,0] * (1 - self.O[1,0]) * self.Wh[1]
        temp *= self.O[0,1] * (1 - self.O[0,1]) * self.Wh[0]
        temp += 1
        temp *= self.O[2,1] * (1 - self.O[2,1])
        self.H_b[1,1] = temp


        # ------------------ Update H_Wh ------------------ #
        # Update H_Wh[0,0]
        temp = self.O[2,0] * (1 - self.O[2,0]) * self.O[1,1]
        self.H_Wh[0,0] = temp

        # Update H_Wh[0,1]
        temp = self.O[2,1] * (1 - self.O[2,1]) * self.Wh[1]
        temp *= self.O[1,0] * (1 - self.O[1,0]) * self.O[0,1]
        self.H_Wh[0,1] = temp

        # Update H_Wh[1,0]
        temp = self.O[2,0] * (1 - self.O[2,0]) * self.Wh[0]
        temp *= self.O[1,1] * (1 - self.O[1,1]) * self.O[0,0]
        self.H_Wh[1,0] = temp

        # Update H_Wh[1,1]
        temp = self.O[2,1] * (1 - self.O[2,1]) * self.O[1,0]
        self.H_Wh[1,1] = temp

    # Routine to set all del terms equal to zero
    def reset_del(self):
        self.del_Wi = np.zeros((self.n, self.i))
        self.del_b = np.zeros((self.n, self.i))
        self.del_Wh = np.zeros((self.n, self.i))

    # Routine for computing error at a single data point
    def update_Err(self, t_row, target):
        for j in range(self.n):
            self.Err[j] = target[t_row, j] - self.O[self.t - 1,j]

    # Routine for computing total error
    def total_Err(self, n_rows, feature, target):
        temp = 0

        # Compute error for all data points in training set
        for j in range(n_rows):
            for k in range(self.t):
                self.update_O(feature[j], k)
    
            self.update_Err(j, target)
            temp += 0.5 * (self.Err[0]**2 + self.Err[1]**2)

        return temp

    # Forward propagation of nth row of feature
    def forward(self, f_row, feature):
        # Perform forward propogation through recurrent neural network
        for j in range(self.t):
            self.update_O(feature[f_row], j)

    # Routine to perform backpropagation
    def backward(self, n_row, feature, target):
        # Compute Error terms for current data point
        self.update_Err(n_row, target)

        # Compute Hessian values for current data point
        self.update_H(n_row, feature)

        # Update del_Wi, del_b, and del_Wh
        for j in range(self.n):
            # self.H_Wi[j,0]*self.Err[0] + self.H_Wi[j,1]*self.Err[1]
            self.del_Wi[j,0] += np.dot(self.H_Wi[j].T, self.Err)
            # self.H_b[j,0]*self.Err[0] + self.H_b[j,1]*self.Err[1]
            self.del_b[j,0] += np.dot(self.H_b[j].T, self.Err)
            # self.H_Wh[j,0]*self.Err[0] + self.H_Wh[j,1]*self.Err[1]
            self.del_Wh[j,0] += np.dot(self.H_Wh[j].T, self.Err)

    # Routine to update weights and biases
    def update_W(self, eta):
        # Perform update of weights and bias by row
        for j in range(self.n):
            self.Wi[j,0] += eta * self.del_Wi[j,0]
            self.b[j,0] += eta * self.del_b[j,0]
            self.Wh[j,0] += eta * self.del_Wh[j,0]

    # Routine to run batch gradient descent
    def batch_grad_descent(self, eta, batch, feature, target):
        for j in batch:
            # Perform forward propagation
            self.forward(j, feature)

            # Perform backward propagation
            self.backward(j, feature, target)

        # Update weights
        self.update_W(eta)

# Main function to train Recurrent Neural Network
def main():
    # Initialize parameters            
    p = Parm(1e-2, .1, 200, 200)

    # Initialize data
    d = Data(2, 2, 'q3data.csv')

    # Initialize recurrent neural network
    N = Rnn(2, 1, 3, False)

    # Train the recurrent neural network on the data over 100 epochs
    for j in range(50):
        # Reset the values in the batch
        p.reset_batch()

        # Reset the values in del
        N.reset_del()

        # Perform the batch stochastic gradient descent
        N.batch_grad_descent(p.eta, p.batch, d.feature, d.target)

        # Compute the total error using the new weights
        err = N.total_Err(p.rows, d.feature, d.target)

        # Check if error is less than the previously smallest error value
        if (err < p.min_err):
            # Update the min_err parameter
            p.update_min_err(err)

            # Update the final weights Wi, b, and Wh
            N.Wi_final = np.copy(N.Wi)
            N.b_final = np.copy(N.b)
            N.Wh_final = np.copy(N.Wh)

            # If error is less than tolerance then exit training 
            if (err < p.tol):
                break

    print("\n ------------------------- Final Weights -------------------------\n")
    print("  w_hat = [[%lg], [%lg]]\n" %(N.Wi_final[0,0], N.Wi_final[1,0]))
    print("  bias = [[%lg], [%lg]]\n" %(N.b_final[0,0], N.b_final[1,0]))
    print("  w = [[%lg], [%lg]]" %(N.Wh_final[0,0], N.Wh_final[1,0]))
    print("\n -------------------------- Total Error --------------------------\n")
    print("  0.5 * || t - o ||^2 = %lg\n" %p.min_err)

# Run main program
main()
