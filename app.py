import argparse
import numpy as np
import pandas as pd
import sklearn.model_selection as skms
from collections import Counter
import matplotlib.pyplot as plt


class Dpreparer:
    @staticmethod
    # Preparing data to further work
    def prepare_data(data, splitter, split_ratio):
        try:
            data = pd.read_table(data, header=None, sep=splitter)
            train_data, test_data = skms.train_test_split(data, random_state=47, train_size=split_ratio)
        except FileNotFoundError:
            print(FileNotFoundError)
        finally:
            return train_data.to_numpy(), test_data.to_numpy()


class Knn:
    # Returns list of vectors with their classified class
    @staticmethod
    def classify(train_data, test_data, index, k):
        classified = []
        try:
            for test_vector in test_data:
                dsc = []
                for train_vector in train_data:
                    dsc.append(
                        [Knn._manhattan_distance(test_vector[:index:], train_vector[:index:]), train_vector[index]])
                nearest = sorted(dsc, key=lambda x: x[0])[:k]
                nearest = np.delete(nearest, 0, axis=1)
                nearest = np.concatenate(nearest, axis=0)
                classified.append([test_vector, Knn._vote(nearest)])
        except TypeError:
            print('Wrong row splitter')
        finally:
            return classified

    # Returns Manhattan distance betwen two vectors
    @staticmethod
    def _manhattan_distance(vector1, vector2):
        dst = np.sum(np.abs(vector1 - vector2))
        return dst

    # Returns most frequent class
    @staticmethod
    def _vote(nearest):
        count = Counter(nearest)
        vote = count.most_common()[0][0]
        return vote

    # Returns accuracy of classified data
    @staticmethod
    def accuracy(test_data, classified, index):
        acc = 0
        for i in range(len(test_data)):
            if test_data[i][index] == classified[i][1]:
                acc += 1
        acc /= len(test_data)
        return acc


# Creating graph - ugly written for iris data set :)
def crt_plt(classified, axis1, axis2):
    xs1, xs2, xs3 = [], [], []
    ys1, ys2, ys3 = [], [], []
    for vec in classified:
        if vec[1] == 'Iris-virginica':
            xs1.append(vec[0][axis1])
            ys1.append(vec[0][axis2])
        elif vec[1] == 'Iris-versicolor':
            xs2.append(vec[0][axis1])
            ys2.append(vec[0][axis2])
        elif vec[1] == 'Iris-setosa':
            xs3.append(vec[0][axis1])
            ys3.append(vec[0][axis2])
    plt.plot(xs1, ys1, 'bo', color='darkorange')
    plt.plot(xs2, ys2, 'bo', color='darkgreen')
    plt.plot(xs3, ys3, 'bo', color='darkblue')
    plt.savefig('graph.png')


# Main body
def main(data, splitter=';', index=0, k=3):
    print('Preparing data')
    train_data, test_data = Dpreparer.prepare_data(data, splitter, split_ratio=0.75)
    print('Classifying')
    classified = Knn.classify(train_data, test_data, index, k)
    if classified:
        accuracy = Knn.accuracy(test_data, classified, index)
        print('Accuracy = ' + str(accuracy))
        crt_plt(classified, axis1=0, axis2=1)
    else:
        print('Classifying failed')


# Parsing arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='KNN algorithm using manhattan distance')
    parser.add_argument('-d', '--data', type=str, required=True, help='Data matrix')
    parser.add_argument('-s', '--splitter', type=str, required=True, help='Row splitter od data')
    parser.add_argument('-i', '--index', type=int, required=True, help='Index of decisional attribute')
    parser.add_argument('-k', '--k', type=int, required=True, help='K value')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.data, args.splitter, args.index, args.k)
