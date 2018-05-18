# %%
import sys
import os
sys.path.append('/Users/aric/cs231n/assignment1')
print(sys.path)
# %%
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
dists = classifier.compute_distances_two_loops(x_test)
print(dists)

# %%
plt.imshow(dists, interpolation='none')
plt.show()


# %%
def get_accuracy(k):
    y_test_pred = classifier.predict_labels(dists, k)
    # print(y_test_pred, y_test)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct)/num_test
    # print('%d / %d correct => accuracy: %f' % (
#     num_correct,
#     num_test,
#     accuracy))

    return accuracy


x = np.arange(1, 50)
y = list(map(lambda x: get_accuracy(x), x))
plt.scatter(x, y)
plt.show()

# %%
dists_one = classifier.compute_distances_one_loop(x_test)
difference = np.linalg.norm(dists-dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# %%
M = np.dot(x_test, x_train.T)
print(M.shape)
te = np.square(x_test).sum(axis=1)
tr = np.square(x_train).sum(axis=1)
print(te.shape, tr.shape)
print((M+tr+te.T).shape)
