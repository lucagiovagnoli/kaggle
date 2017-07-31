import math
import matplotlib.pyplot as plt
import numpy
import pandas
import pydotplus

from sklearn import tree
from sklearn import preprocessing


def load_csv_to_numpy(filename):
    dataset = pandas.read_csv(filename)
    dataset = dataset.fillna(0)

    # from numpy import genfromtxt
    # my_data = genfromtxt(
        # filename,
        # delimiter=',',
        # names=True,
        # missing_values='',
        # filling_values=0.0,
    # )

    npy_data = numpy.copy(dataset.values)
    columns_labels = dataset.columns.values

    return npy_data, columns_labels


def preproc(matrix, columns_to_encode):
    X = numpy.copy(matrix)
    
    for column_to_encode in columns_to_encode:
	label_encoder = preprocessing.LabelEncoder()
	label_encoder.fit(X[:,column_to_encode])
	encoded_labels = label_encoder.transform(X[:,column_to_encode])
	X[:, column_to_encode] = encoded_labels

    return X


def train_model(npy_data):

    Y = npy_data[:,1].astype(int)
    Y.reshape(1, len(Y))
    X = npy_data[:,2:]

    X = preproc(X, [1, 2, 6, 8, 9])

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    return clf

def load_results():
    passenger_to_survival = {}
    npy_data, _ = load_csv_to_numpy("gender_submission.csv")

    for record in npy_data:
        passenger_to_survival[record[0]] = record[1]

    return passenger_to_survival

npy_data, column_labels = load_csv_to_numpy("train.csv")
clf = train_model(npy_data)

npy_test, _ = load_csv_to_numpy("test.csv")
X_test = preproc(npy_test, [2, 3, 7, 9, 10])


passenger_to_survival = load_results()

count = 0
for record in X_test:
    if clf.predict(record.reshape(1, len(record))[:, 1:]) == passenger_to_survival[record[0]]:
        count += 1

print(float(count)/float(len(passenger_to_survival)))

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=column_labels[1:])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")

