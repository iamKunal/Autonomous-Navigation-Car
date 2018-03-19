#!/usr/bin/python2


import numpy as np
from sklearn.model_selection import train_test_split
import glob
import time
import cv2


train_data_file = sorted(glob.glob('train_data/*.npz'))[-1]


with np.load(train_data_file) as data:
    train_data = data['data'].astype(np.float32)
    train_labels = data['labels'].astype(np.float32)


td = train_data[1:, :]
tl = train_labels[1:, :]


# print tl

assert td.shape[0] == tl.shape[0]


train, test, train_labels, test_labels = train_test_split(td, tl, test_size=0.2)



layer_sizes = np.int32([td.shape[1], 40, tl.shape[1]])

# model = cv2.ml.ANN_MLP_create()
# model.setLayerSizes(layer_sizes)
# # # criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5, 0.0001)
# # criteria = (cv2.TERM_CRITERIA_EPS,1000, 0.01)
# # model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
# # animals_net.setTermCriteria(( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ))
# # model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
# # model.setBackpropWeightScale(0.0001)
# # # model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)

# # model.setBackpropMomentumScale(0.1)


# # model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
# # model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
# # model.setTermCriteria(( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ))

model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(layer_sizes)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setBackpropMomentumScale(0.1)
model.setBackpropWeightScale(0.001)
model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.1))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)


print 'Training MLP ...'

num_iter = model.train(train, cv2.ml.ROW_SAMPLE, train_labels)


print 'Ran for %d iterations' % num_iter

# train data
ret_0, resp_0 = model.predict(train)
prediction_0 = resp_0.argmax(-1)
true_labels_0 = train_labels.argmax(-1)

train_rate = np.mean(prediction_0 == true_labels_0)
print 'Train accuracy: ', "{0:.2f}%".format(train_rate * 100)

# test data
ret_1, resp_1 = model.predict(test)
prediction_1 = resp_1.argmax(-1)
true_labels_1 = test_labels.argmax(-1)

test_rate = np.mean(prediction_1 == true_labels_1)
print 'Test accuracy: ', "{0:.2f}%".format(test_rate * 100)

# save model
model.save('trained_model/%d.xml' % (int(time.time())))