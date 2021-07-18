import tensorflow as tf
import numpy as np
import cv2

width = 640
height = 480
threshold = 0.65
cameraNo = 0


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def test(name):
    model = tf.keras.models.load_model('trained_model.h5')
    img = cv2.imread('Upload/' + name)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)
    # classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    # probVal = np.amax(predictions)
    a = np.amax(predictions) * 100
    x = "%.2f" % round(a, 2)
    return x


# print(test('test.jpg'))