# %%
import sys
import os
sys.path.append('/Users/aric/cs231n/assignment1')

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

% load_ext autoreload
%autoreload 2
print(sys.path)
# %%
cifar10_dir = '/Users/aric/cs231n/assignment1/cs231n/datasets/cifar-10-batches-py'
x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)

print(f'Train{x_train.shape}')

# %%
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
sample_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(
        idxs,
        sample_per_class,
        replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(
            sample_per_class,
            num_classes,
            plt_idx
        )
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show

# %%
num_training = 5000
mask = range(num_training)
x_train = x_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
x_test = x_test[mask]
y_test = y_test[mask]

# %%
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print(x_train.shape, x_test.shape)

# %%
from cs231n.classifiers import KNearestNeighbor

classifier = KNearestNeighbor()
classifier.train(x_train, y_train)
print('x')

# %%
dists = classifier.compute_distances_no_loops(x_test)
print(dists)

# %%
plt.imshow(dists, interpolation='none')
plt.show()


# %%
def get_accuracy(classifier, x_test, y_test, k):
    y_test_pred = classifier.predict_labels(x_test, k)
    # print(y_test_pred, y_test)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct)/num_test
    # print('%d / %d correct => accuracy: %f' % (
#     num_correct,
#     num_test,
#     accuracy))

    return accuracy


# %%
x = np.arange(1, 50)
y = list(map(lambda x: get_accuracy(classifier, x_test, y_test, x), x))
plt.scatter(x, y)
plt.show()


# %%
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' %
      (num_correct, num_test, accuracy))
# %%
dists_one = classifier.compute_distances_one_loop(x_test)
difference = np.linalg.norm(dists-dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# %%
dists_two = classifier.compute_distances_no_loops(x_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')


# %%
import time


def time_function(f, *args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


two_loop_time = time_function(
    classifier.compute_distances_two_loops,
    x_test
)
one_loop_time = time_function(
    classifier.compute_distances_one_loop,
    x_test
)
no_loop_time = time_function(
    classifier.compute_distances_no_loops,
    x_test
)
print('Two loop version took %f seconds' % two_loop_time)

print('One loop version took %f seconds' % one_loop_time)

print('No loop version took %f seconds' % no_loop_time)

# %%
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
x_train_folds = []
y_train_folds = []
# x_train_folds = np.array_split(x_train, num_folds)
# y_train_folds = np.array_split(y_train, num_folds)
y_train_ = y_train.reshape(-1, 1)
X_train_folds, y_train_folds = np.array_split(
    x_train, 5), np.array_split(y_train_, 5)

# %%
k_to_accuracies = {}
cross_classifiers = KNearestNeighbor()
for k in k_choices:
    k_to_accuracies.setdefault(k, [])
for i in range(num_folds):
    x_val = X_train_folds[i]
    y_val = y_train_folds[i]
    x_cross_train = np.delete(X_train_folds, i, 0)
    x_cross_train = np.concatenate(x_cross_train)
    y_cross_train = np.delete(y_train_folds, i, 0)
    y_cross_train = np.concatenate(y_cross_train)
    cross_classifiers.train(x_cross_train, y_cross_train)
    for k_ in k_choices:
        accuracy = get_accuracy(
            cross_classifiers,
            x_val,
            y_val,
            k_)
        k_to_accuracies[k_].append(accuracy)
print(k_to_accuracies)

# %%
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
# %%
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)
accuracies_mean = np.array([np.mean(v)
                            for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v)
                           for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# %%
best_k = 1

classifier = KNearestNeighbor()
classifier.train(x_train, y_train)
y_test_pred = classifier.predict(x_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' %
      (num_correct, num_test, accuracy))
