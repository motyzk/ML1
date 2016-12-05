import matplotlib.pyplot as plt
import numpy.linalg
import numpy.random
from sklearn.datasets import fetch_mldata
#import pdb
import operator

#retrieving from MNIST and setting train and test sets
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :]
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :]
test_labels = labels[idx[10000:]]

#constants
INDEX = 0
LABEL = 1


def knn(images_set, labels_vec, query_image, k):
    query_image_vec = numpy.array(query_image, float)
    dist_vec = sorted(((numpy.linalg.norm(numpy.array(image, float) - query_image_vec), label)
                      for image, label in zip(images_set, labels_vec)), key=operator.itemgetter(INDEX))
    # plt.imshow(numpy.concatenate([query_image_vec.reshape((28,28))]+[vec.reshape((28,28)) for _, _, vec
    #                                                                  in dist_vec[:k]], axis=1))
    # plt.title(str(digit_histogram))
    # plt.waitforbuttonpress()
    # generate a histogram and return the index of max
    return numpy.argmax(numpy.bincount(zip(*dist_vec[:k])[LABEL]))


def b():
    labeling_the_test_set = [knn(train[:1000], train_labels[:1000], test_image, 10) for test_image in test]
    return sum(1 if my_label == real_label else 0
               for my_label, real_label in zip(labeling_the_test_set, test_labels[:1000])) / 1000.0


# def test_knn():
#     if 9 != knn(images_set=[[3, 3], [1, 1], [-3, -3]],
#                 labels_vec=[9, 9, 0],
#                 query_image=[2, 2],
#                 k=2):
#         print "1. :("
#     if 5 != knn(images_set=[[-3, -3], [-1, -1], [3, 3]],
#                 labels_vec=[5, 5, 8],
#                 query_image=[-2, -2],
#                 k=2):
#         print "2. :("
#     if 3 != knn(images_set=[[3, 3], [1, 1], [-3, -3], [3, 3]],
#                 labels_vec=[4, 3, 9, 3],
#                 query_image=[2, 2],
#                 k=3):
#         print "3. :("
#     if 0 != knn(images_set=[[3, 3], [1, 1], [-3, -3], [3, 3]],
#                 labels_vec=[0, 0, 1, 0],
#                 query_image=[2, 2],
#                 k=3):
#         print "4. :("
#     print "testing KNN complete"
#
#
# def testing():
#     test_knn()
#     print b()
# testing()
