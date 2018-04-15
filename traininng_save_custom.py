from ANN import *
import time
import cv2
import glob

if __name__ == '__main__':
    train_data_file = sorted(glob.glob('train_data/*.npz'))[-1]

    with np.load(train_data_file) as data:
        train_data = data['data'].astype(np.float32)
        train_labels = data['labels'].astype(np.float32)

    td = train_data[1:, :]
    tl = train_labels[1:, :]

    model = NeuralNetwork()
    layer_sizes = np.int32([td.shape[1], 40, tl.shape[1]])

    model.set_layer_sizes(layer_sizes)
    model.forward_propagate([1]*20000)
    # model.train(train_data, train_labels)