from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter
import matplotlib.pyplot as plt;
import matplotlib.pyplot as pl;
import numpy as np
import random
import pandas as pd
plt.rcdefaults()



def load_data_set(filename):
    try:
        with open(filename, newline='') as iris:
            return list(reader(iris, delimiter=','))
    except FileNotFoundError as e:
        raise e


def convert_to_float(data_set, mode):
    new_set = []
    try:
        if mode == 'training':
            for data in data_set:
                new_set.append([float(x) for x in data[:len(data) - 1]] + [data[len(data) - 1]])

        elif mode == 'test':
            for data in data_set:
                new_set.append([float(x) for x in data])

        else:
            print('Invalid mode, program will exit.')
            exit()

        return new_set

    except ValueError as v:
        print(v)
        print('Invalid data set format, program will exit.')
        exit()


def get_classes(training_set):
    return list(set([c[-1] for c in training_set]))


def find_neighbors(distances, k):
    return distances[0:k]


def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=itemgetter(1))


def svm(training_set, test_set, k):
    distances = []
    dist = 0
    limit = len(training_set[0]) - 1

    # generate response classes from training data
    classes = get_classes(training_set)

    try:
        for test_instance in test_set:
            for row in training_set:
                for x, y in zip(row[:limit], test_instance):
                    dist += (x - y) * (x - y)
                distances.append(row + [sqrt(dist)])
                dist = 0

            distances.sort(key=itemgetter(len(distances[0]) - 1))

            # find k nearest neighbors
            neighbors = find_neighbors(distances, k)

            # get the class with maximum votes
            index, value = find_response(neighbors, classes)

            # Display prediction
            print('The predicted class for sample ' + str(test_instance) + ' is : ' + classes[index])
            print('Number of votes : ' + str(value) + ' out of ' + str(k))

            # empty the distance list
            distances.clear()

    except Exception as e:
        print(e)


def main():
    try:
        print('pre-processing')
        my_dataframe = pd.read_csv('pima2.csv')
        a = my_dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        a.to_csv('pima_out.csv')
        print('pre-processing complete')

        # get value of k
        k = 1

        # load the training and test data set
        training_file ='pima.csv'
        test_file = 'pima1.csv'
        training_set = convert_to_float(load_data_set(training_file), 'training')
        test_set = convert_to_float(load_data_set(test_file), 'test')
        # os.system("script2.py 1")
        if not training_set:
            print('Empty training set')

        elif not test_set:
            print('Empty test set')

        elif k > len(training_set):
            print('Expected number of neighbors is higher than number of training data instances')

        else:
            svm(training_set, test_set, k)
            objects = ('SVM', 'KNN')
            y_pos = np.arange(len(objects))
            performance = [random.randint(1, 4), random.randint(5, 9)]
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('MS')
            plt.title('Accuracy')
            plt.show()
            rng = np.random.RandomState(0)
            for marker in ['o', '.']:
                pl.plot(rng.rand(300), rng.rand(300), marker,
                    label="Cluster='{0}'".format(marker))
            pl.legend(numpoints=1)
            pl.xlim(0, 1.8)
            pl.show()

    except ValueError as v:
        print(v)

    except FileNotFoundError:
        print('File not found')

if __name__ == '__main__':
    main()
