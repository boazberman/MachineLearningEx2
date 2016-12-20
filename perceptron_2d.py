import math
import sys

import matplotlib.pyplot as plt
import numpy as np


def import_data(dataSetFilePath):
    data = np.loadtxt(dataSetFilePath, delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]

    return x, y


def calc_r(x):
    r = 0
    for idx, xi in enumerate(x):
        square_size = sum(x ** 2 for x in xi.tolist())
        if math.sqrt(square_size) > r:
            r = math.sqrt(square_size)

    return r


def gamma(x, y, normalize_w):
    gama = sys.float_info.max
    for idx, xi in enumerate(x):
        temp = np.dot([1] + xi.tolist(), normalize_w) * y[idx]
        if temp < gama:
            gama = temp * y[idx]

    return gama


def perceptron(x, y, learningRate=1):
    mistakes, iterations = 0, 0
    should_continue = True
    w = [0] * (x[0].size + 1)
    while should_continue:
        should_continue = False
        iterations += 1
        for idx, xi in enumerate(x):
            xi_list = [1] + xi.tolist()
            if np.dot(xi_list, w) * y[idx] <= 0:
                w = [(y[idx] * xi_list[w_index] * learningRate) + elem for w_index, elem in
                     enumerate(w)]
                should_continue = True
                mistakes += 1

    return w, mistakes, iterations


def split_by_sign(y):
    pos_index, neg_index = [], []
    for i, yi in enumerate(y):
        pos_index.append(i) if yi > 0 else neg_index.append(i)

    return pos_index, neg_index


def plot_data(x, y):
    pos_index, neg_index = split_by_sign(y)
    plt.plot([x[i][0] for i in pos_index], [x[i][1] for i in pos_index], 'bo')
    plt.plot([x[i][0] for i in neg_index], [x[i][1] for i in neg_index], 'ro')
    plt.show()


def normalize(vector):
    return [float(x / math.sqrt(sum(x * x for x in vector))) for x in vector]


def calc_mistake_bound():
    return (calc_r(x) / gamma(x, y, normalize(w))) ** 2


if __name__ == '__main__':
    x, y = import_data(sys.argv[1])
    # plot_data(x, y)
    w, mistakes, iterations = perceptron(x, y)
    # plot_separation_hyperplane(w,x)

    with open("output.txt", 'w') as outputFile:
        outputFile.write("output1: %s\n" % w)
        outputFile.write("output2: %s\n" % mistakes)
        outputFile.write("output3: %s\n" % iterations)
        outputFile.write("output4: %s\n" % calc_mistake_bound())
