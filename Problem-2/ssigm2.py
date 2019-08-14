import numpy as np
import pylab
import matplotlib.pyplot as plt

# Define sigmoid function
def sigm(x):
    # To prevent overflow, we use the following stable implementation
    return 1./(1. + np.exp(-x))

# Define function for generating target data from feature data
def gen_target(data, n_rows, a, b, r):
    r2 = r*r
    target = np.zeros(n_rows)
    for i in range(n_rows):
        temp = (data[i, 0] - a)**2 + (data[i, 1] - b)**2
        if temp < r2:
            target[i] += 1

    return target

# Define class to contain parameters for neural network
class Parm:
    # Initialize parameters
    def __init__(self, eta, b_size, data_rows, data_cols):
        # Initialize learning rate
        self.eta = eta

        # Initialize batch size
        self.batch_size = b_size

        # Initialize data row and column sizes
        self.rows = data_rows
        self.cols = data_cols

        # Initialize batch
        self.batch = np.random.randint(0, self.rows, self.batch_size)

    # Modify values in batch
    def shuffle(self):
        self.batch = np.random.randint(0, self.rows, self.batch_size)

# Define class to contain weights for neural network
class Weights:
    # Initialize parameters
    def __init__(self, data_cols, n_hidden):
        # Initialize number of hidden nodes
        self.n_hidden = n_hidden

        # Initialize original input bias terms 
        self.bi_orig = np.random.uniform(-1, 1, n_hidden)

        # Initialize original hidden bias term 
        self.bh_orig = np.random.uniform(-1, 1)

        # Initialize original input weights
        self.Wi_orig = np.random.uniform(-.1, .1, (n_hidden, data_cols))

        # Initialize original hidden weights
        self.Wh_orig = np.random.uniform(-.1, .1, n_hidden)

        # Initialize input bias terms 
        self.bi = np.copy(self.bi_orig)

        # Initialize hidden bias term 
        self.bh = np.copy(self.bh_orig)

        # Initialize input weights
        self.Wi = np.copy(self.Wi_orig)

        # Initialize hidden weights
        self.Wh = np.copy(self.Wh_orig)

    # Routine for setting weights and bias terms to original values
    def reset_weights(self):
        self.bi = np.copy(self.bi_orig)
        self.bh = np.copy(self.bh_orig)
        self.Wi = np.copy(self.Wi_orig)
        self.Wh = np.copy(self.Wh_orig)

class Network:
    # Initialize parameters
    def __init__(self, eta, b_size, data_rows, data_cols, n_hidden):
        self.w = Weights(data_cols, n_hidden)
        self.p = Parm(eta, b_size, data_rows, data_cols)

    # Define function for resetting weights
    def reset_w(self):
        self.w.reset_weights()

    # Define function to generate output of neural network
    def h_out(self, feature):
        # Compute value at which to evaluate sigmoid function
        x = np.dot(self.w.Wi, feature) + self.w.bi
        return sigm(x)

    def o_out(self, x):
        # Compute value at which to evaluate sigmoid function
        temp = np.dot(self.w.Wh, x) + self.w.bh
        return sigm(temp)

    # Define error function for neural network
    def err(self, target, output):
        return target - output

    # Define total error function for neural network
    def tot_err(self, feature, target, test_size):
        temp = 0

        for i in range(test_size):
            o_h = self.h_out(feature[i])
            o = self.o_out(o_h)
            temp += 0.5 * (target[i] - o)**2

        print("  Total error = ", temp)

    def update_Wh (self, Err, o, o_h):
        # Compute gradient of Error with respect to hidden bias
        temp = self.p.eta * Err * o * (1 - o)
        # Update hidden bias term
        self.w.bh += temp

        # Compute gradient of Error with respect to hidden weights
        update = o_h * temp 
        # Update hidden weights
        self.w.Wh += update

    def update_Wi (self, feature, Err, o, o_h):
        temp = self.p.eta * Err * o * (1 - o) * self.w.Wh * o_h * (1 - o_h)
        self.w.bi = np.add(self.w.bi, temp)

        for j in range(self.w.n_hidden):
            update = feature * temp[j]

            self.w.Wi[j] = np.add(self.w.Wi[j], update)

    # Define function for training neural network on data set
    def train(self, feature, target):
        num_err = 0

        for i in range(self.p.rows):
            o_h = self.h_out(feature[i])
            o = self.o_out(o_h)
            Err = self.err(target[i], o)

            self.update_Wh(Err, o, o_h)
            self.update_Wi(feature[i], Err, o, o_h)

    # Define function for testing neural network on data set
    def test(self, feature, target, test_size, data_title):
        num_err = 0

        for i in range(test_size):
            o_h = self.h_out(feature[i])
            o = self.o_out(o_h)
            Err = self.err(target[i], o)
            
            if(o >= 0.5):
                o = 1
            else:
                o = 0

            if(target[i] != o):
                num_err += 1  

        print("  Number of errors from", data_title, ":", num_err)

    # Define function for plotting test results from Nerual Network on test set
    def plot_test(self, feature, target, test_size, plot_title):
        x_no = []
        x_yes = []
        y_no = []
        y_yes = []

        for i in range(test_size):
            o_h = self.h_out(feature[i])
            o = self.o_out(o_h)
            
            # Store predictions based on model developed by Neural Network
            if (o >= 0.5):
                x_yes.append(feature[i, 0])
                y_yes.append(feature[i, 1])
            else:
                x_no.append(feature[i, 0])
                y_no.append(feature[i, 1])

        # Commands for plotting classified data points
        ax = plt.gca()
        ax.cla()             # Clear the grid for fresh plot
        plt.plot(x_no[:], y_no[:], 'r^', markersize=6, lw=5)
        plt.plot(x_yes[:], y_yes[:], 'go', markersize=6, lw=3)
        circle1 = plt.Circle((0.5, 0.6), 0.4, color='k', lw=3, fill=False)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.add_artist(circle1)
        # set and label axes, set title, and legend
        plt.axis([0, 1, 0, 1])
        plt.title(plot_title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        plot_title.__add__('.png')
        plt.savefig(plot_title)

# Main function
def main(epochs):
    # Initialize Neural Network
    N = Network(1, 5, 100, 2, 10)

    # Generate training data
    train_feature = np.random.uniform(0, 1, (N.p.rows, N.p.cols))
    train_target = gen_target(train_feature, N.p.rows, 0.5, 0.6, 0.4)

    # Generate testing data
    testing_size = 100
    test_feature = np.random.uniform(0, 1, (testing_size, N.p.cols))
    test_target = gen_target(test_feature, testing_size, 0.5, 0.6, 0.4)

    # Print some statistics prior to training
    print("\n --------------- Statistics prior to training ---------------")
    N.test(train_feature, train_target, N.p.rows, 
          'training data prior to training')
    N.tot_err(train_feature, train_target, N.p.rows)

    N.test(test_feature, test_target, testing_size, 
          'testing data prior to training')
    N.tot_err(test_feature, test_target, testing_size)

    # Traing and test over epochs provided as input
    for i in range(epochs.size):
        # Reset weights for next training interval
        N.reset_w()

        for j in range(epochs[i]):
            N.train(train_feature, train_target)

        # Print some statistics after training
        print("\n -------------- After training over %d epochs --------------" 
              %epochs[i])
    
        N.test(train_feature, train_target, N.p.rows, 
              'training data after training')
        N.tot_err(train_feature, train_target, N.p.rows)

        N.test(test_feature, test_target, testing_size, 
              'testing data after training')
        N.tot_err(test_feature, test_target, testing_size)

        N.plot_test(train_feature, train_target, N.p.rows, 
                    'Plot of Classification of Training Data '
                    'over %d epochs' %epochs[i])
        N.plot_test(test_feature, test_target, testing_size, 
                    'Plot of Classification of Testing Data '
                    'over %d epochs' %epochs[i])

    print("\n")

# Run main function
main(np.array([100, 250, 500, 2500, 5000]))

