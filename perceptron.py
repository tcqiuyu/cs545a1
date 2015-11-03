__author__ = 'Qiu Yu'

import numpy as np
from matplotlib import pyplot as plt


class Perceptron:
    def __init__(self, max_iterations=400, learning_rate=0.2, bias=0.0):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.bias = bias

    def fit(self, w, X, y, data_name):
        """
        Train a classifier using the perceptron training algorithm.
        After training the attribute 'w' will contain the perceptron weight vector.

        :param X: ndarray, shape (num_examples, n_features)
         Training data.

        :param y: ndarray, shape (n_examples,)
         Array of labels.
        """
        self.print_info("Basic Perceptron", data_name, X)
        self.w = w
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations):
            converged = True
            for i in range(len(X)):
                if self.discriminant(X[i], y[i]) <= 0:
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
            iterations += 1
            self.converged = converged
            # plot_data(X, y, self.w)
        print 'Result:'
        if converged:
            print '\tconverged in %d iterations' % iterations
        else:
            print '\tnot converged in %d iterations' % iterations
        print '\tw = ' + self.w.__str__()
        misclassification, E_in = self.calculate_E(X, y)
        print "\tThere are %d misclassfication in training set ( %d in total )" % (misclassification, X.shape[0])
        print "\tE_in = %.2f" % E_in
        print "\tAccuracy = %.2f%%" % ((1 - E_in) * 100)

    def discriminant(self, x, y):
        return y * (np.dot(self.w, x) + self.bias)

    def calculate_E(self, X, y):
        scores = np.dot(X, self.w)
        result = np.sign(scores)
        E = 0.0
        for i in range(0, result.shape[0]):
            if result[i] != y[i]:
                E += 1
        misclassification = E
        E = E / result.shape[0]
        return int(misclassification), E

    def predict(self, X, y):
        """
        make predictions using a trained linear classifier

        :param x: ndarray, shape (num_examples, n_features)
        Training data
        """
        print '.....................Predicting Process.......................'
        print 'Testing set size = %d' % X.shape[0]
        print 'Result:'
        misclassification, E_out = self.calculate_E(X, y)
        print "\tThere are %d misclassfication in test case ( %d in total )" % (misclassification, X.shape[0])
        print "\tE_out = %.2f" % E_out
        print "\tAccuracy = %.2f%%\n\n" % ((1 - E_out) * 100)
        return E_out

    def print_info(self, algo_name, data_name, X):
        print '<--------------------------------- %s training starts on %s (Bias = %d) --------------------------------->' % (
            algo_name, data_name, self.bias)
        print '.......................Training Process.......................'
        print 'Training set size = %d' % X.shape[0]


class PocketPerceptron(Perceptron):
    def fit(self, w, X, y, data_name):
        self.print_info("Perceptron with pocket algorithm", data_name, X)
        self.w = w
        self.w_pocket = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        E_pocket = -1
        iter_pocket = 0
        w_pocket = []
        while (not converged and iterations < self.max_iterations):
            converged = True
            E_in = 0
            for i in range(len(X)):
                disciminant = self.discriminant(X[i], y[i])
                if disciminant <= 0:
                    E_in += abs(disciminant)
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
            iterations += 1
            if E_pocket == -1 or E_in < E_pocket:
                E_pocket = E_in
                iter_pocket = iterations
                w_pocket = self.w
            self.converged = converged
        print 'Result:'
        if converged:
            print '\tconverged in %d iterations' % iterations
            print '\tw = ' + self.w.__str__()
        else:
            print '\tnot converged in %d iterations' % iterations
            print '\tbest w in iteration %d' % iter_pocket
            print '\tw = ' + w_pocket.__str__()
            self.w = w_pocket
        misclassification, E_in = self.calculate_E(X, y)
        print "\tThere are %d misclassfication in training set ( %d in total )" % (misclassification, X.shape[0])
        print "\tE_in = %.2f" % E_in
        print "Accuracy = %.2f" % ((1 - E_in) * 100)


class ModifiedPerceptron(Perceptron):
    def fit(self, w, X, y, data_name):
        self.print_info("Modified Perceptron", data_name, X)
        self.w = w
        self.c = np.random.uniform(0, 1)
        print 'constant c = %f' % self.c
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations):
            converged = True
            j = 0
            lamd = None
            for i in range(len(X)):
                discriminant = self.discriminant(X[i], y[i])
                if discriminant <= self.c * np.linalg.norm(self.w):
                    if lamd == None or discriminant > lamd:
                        lamd = discriminant
                        j = i
                    converged = False
            self.w = self.w + self.learning_rate * y[j] * X[j]
            iterations += 1
        print 'Result:'
        if converged:
            print '\tconverged in %d iterations' % iterations
        else:
            print '\tnot converged in %d iterations' % iterations
        print '\tw = ' + self.w.__str__()
        misclassification, E_in = self.calculate_E(X, y)
        print "\tThere are %d misclassfication in training set ( %d in total )" % (misclassification, X.shape[0])
        print "\tE_in = %.2f" % E_in
        print "Accuracy = %.2f" % ((1 - E_in) * 100)


def plot_data(x, y):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(x, y)
    axes.set_xscale('log', basex=2)
    axes.set_title("Learning Curve")
    axes.set_xlabel("Training set size")
    axes.set_ylabel("Accuracy")
    plt.plot(x, y)
    plt.show()


def divide_set(X, y, test_size):
    random_choice = np.random.choice(X.shape[0], size=test_size, replace=False)
    X_testing_set = X[random_choice, :]
    X_training_set = np.delete(X, random_choice, axis=0)
    y_testing_set = y[random_choice]
    y_training_set = np.delete(y, random_choice, axis=0)
    return X_training_set, X_testing_set, y_training_set, y_testing_set


def process_perceptron(X, y, test_size, p, p_bias, p_pocket, p_modified, data_name):
    w = np.zeros(len(X[0]))
    w_modified = np.random.uniform(-1, 1, len(X[0]))
    X_training, X_testing, y_training, y_testing = divide_set(X, y, test_size)

    p.fit(w, X_training, y_training, data_name)
    p.predict(X_testing, y_testing)
    p_bias.fit(w, X_training, y_training, data_name)
    p_bias.predict(X_testing, y_testing)
    p_pocket.fit(w, X_training, y_training, data_name)
    p_pocket.predict(X_testing, y_testing)
    p_modified.fit(w_modified, X_training, y_training, data_name)
    p_modified.predict(X_testing, y_testing)


def perceptron_normalization(X, y, test_size, p, data_name):
    w = np.zeros(len(X[0]))
    w_modified = np.random.uniform(-1, 1, len(X[0]))
    X_training, X_testing, y_training, y_testing = divide_set(X, y, test_size)
    p.fit(w, X_training, y_training, data_name)
    p.predict(X_testing, y_testing)


def learning_curve(X, y, test_size, p_pocket, starting_size=10, log_base=2):
    X_training, X_testing, y_training, y_testing = divide_set(X, y, test_size)
    learning_step = starting_size
    total_training = X_training.shape[0]
    coord_x = []
    coord_y = []
    while learning_step <= total_training:
        random_choice = np.random.choice(total_training, size=learning_step, replace=False)
        X_training_subset = X_training[random_choice, :]
        y_training_subset = y_training[random_choice]
        w = np.zeros(len(X_training_subset[0]))
        p_pocket.fit(w, X_training_subset, y_training_subset, "Gisette Data (Learning Curve)")
        E_out = p_pocket.predict(X_testing, y_testing)
        coord_x.append(learning_step)
        coord_y.append(1 - E_out)
        learning_step *= log_base
    p_pocket.fit(np.zeros(len(X_training[0])), X_training, y_training, "Gi`sette Data (Learning Curve)")
    E_out = p_pocket.predict(X_testing, y_testing)
    coord_x.append(len(X_training[0]))
    coord_y.append(1 - E_out)
    plot_data(coord_x, coord_y)
    return coord_x, coord_y


def scale(X, type):
    # type 1: normalization
    # type 2 ( or other ): standardization
    def scale_func(col, type):
        if type == 1:
            min = np.min(col)
            max = np.max(col)
            return (2 * col - max - min) / (max - min)
        else:
            mean = np.mean(col)
            std = np.std(col)
            return (col - mean) / std

    output = np.empty(shape=X.shape)
    for i in range(0, X.shape[1]):
        col = X[:, i]
        min = np.min(col)
        max = np.max(col)
        if max == min:
            output[:, i] = np.zeros(len(X.shape[1]))
        else:
            output[:, i] = scale_func(col, type)
    print output, output.shape
    return output


if __name__ == '__main__':
    bias = np.random.random_integers(0, 100)
    p = Perceptron()
    p_bias = Perceptron(bias=bias)
    p_pocket = PocketPerceptron(bias=bias)
    p_modified = ModifiedPerceptron()

    # pre-process heart data
    data_1 = np.genfromtxt("heart.data", delimiter=",", comments="#")
    X_1 = data_1[:, 2:]
    y_1 = data_1[:, 1]
    process_perceptron(X_1, y_1, test_size=100, p=p, p_bias=p_bias, p_pocket=p_pocket, p_modified=p_modified,
                       data_name="Heart Data")

    # pre-process gisette data
    data_2 = np.genfromtxt("gisette_train.data")
    X_2 = data_2
    y_2 = np.genfromtxt("gisette_train.labels")
    process_perceptron(X_2, y_2, test_size=1500, p=p, p_bias=p_bias, p_pocket=p_pocket, p_modified=p_modified,
                       data_name="Gisette Data")
    learning_curve(X_2, y_2, test_size=1500, p_pocket=p_pocket)

    X_1_norm = scale(X_1, 1)
    X_1_stand = scale(X_1, 2)
    perceptron_normalization(X_1_norm, y_1, test_size=100, p=p_bias, data_name="Heart Data (Normalized)")
    perceptron_normalization(X_1_stand, y_1, test_size=100, p=p_bias, data_name="Heart Data (Standardized)")
