import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

MODEL_PATH = Path('app/ml/model')


class CustomModel:
    def __init__(self, path: str):
        self.infer = tf.saved_model.load(path).signatures['serving_default']

    def predict(self, img: np.ndarray):
        infer_dict = self.infer(tf.constant(img, dtype=float))
        keys = infer_dict.keys()
        return int(infer_dict[list(keys)[0]].numpy().argmax(axis=1)[0])

    def preprocess_img(self, img: cv2.Mat):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 100))
        img = img / 255
        img = np.reshape(img, (1, 100, 100, 1))
        return img

    def preprocess_predict(self, img: cv2.Mat):
        return self.predict(self.preprocess_img(img))


model = CustomModel(MODEL_PATH)

