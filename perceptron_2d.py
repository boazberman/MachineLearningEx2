import sys

import matplotlib.pyplot as plt
import numpy as np


def import_data(dataSetFilePath):
    data = np.loadtxt(dataSetFilePath, delimiter=',')


def parse_data(data):
    """
    Returns x, y
    """
    return data[:, :-1], data[:, -1]


def calc_r(x):
    R = 0
    for idx, xi in enumerate(x):
        lst = xi.tolist()
        if (lst[0] * lst[0] + lst[1] * lst[1] > R):
            R = lst[0] * lst[0] + lst[1] * lst[1]

    return R


def calcGama(x, y):
    gama = sys.float_info.max
    for idx, xi in enumerate(x):
        lst = xi.tolist()
        if (y[idx] * lst[0] + y[idx] * lst[1] < gama):
            gama = y[idx] * lst[0] + y[idx] * lst[1]

    return gama


def cartezian_product(x, w):
    sum = 0
    for i, e in enumerate(x):
        sum += e * w[i]
    return sum


def perceptronAlgo(x, y, learningRate=1):
    weight_vector = [0] * (x[0].size + 1)
    input = []
    bias = 1
    mistakes, iterations = 0, 0
    weights_changed = False
    while not weights_changed:
        weights_changed = True
        for idx, xi in enumerate(x):
            iterations += 1
            xi_list = [bias] + xi.tolist()
            if cartezian_product(xi_list, weight_vector) * y[idx] <= 0:
                weight_vector = [y[idx] * xi_list[w_index] * learningRate + elem for w_index, elem in
                                 enumerate(weight_vector)]
                weights_changed = False
                mistakes += 1

    return weight_vector, mistakes, iterations


def split_by_sign(y):
    pos_index, neg_index = [], []
    for i, yi in enumerate(y):
        if yi > 0:
            pos_index.append(i)
        elif yi < 0:
            neg_index.append(i)
    return pos_index, neg_index


def plot_data(x, y):
    pos_index, neg_index = split_by_sign(y)
    plt.plot([x[i][0] for i in pos_index], [x[i][1] for i in pos_index], 'bo')
    plt.plot([x[i][0] for i in neg_index], [x[i][1] for i in neg_index], 'ro')
    plt.show()


def plot_separation_hyperplane(w, original_x):
    """
    This is an attempt to solve 1.(f)
    :rtype: object
    """
    hyperplane = '(%s)*x+1*(%s)' % (w[2] / w[1], w[0] / w[1])
    min_x = min(original_x[::1])
    x = np.array(range(min_x, 10))
    evaled = eval(hyperplane)
    plt.plot(x, evaled)
    plt.show()


if __name__ == '__main__':
    original_data = import_data(sys.argv[1])
    x, y = parse_data(original_data)
    plot_data(x, y)
    w, mistakes, iterations = perceptronAlgo(x, y, 1)
    # plot_separation_hyperplane(w)
    print w, mistakes, iterations

    outputFile = open("output.txt", 'w')
    outputFile.write("output1: " + str(w) + "\n")
    outputFile.write("output2: " + str(mistakes) + "\n")
    outputFile.write("output3: " + str(iterations) + "\n")
    #
    # r = calcR(x)
    # gama = calcGama(x,y)


    # print (r/gama)*(r/gama)
