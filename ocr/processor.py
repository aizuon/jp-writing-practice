import threading

import cv2
import imutils
import numpy as np
from keras.models import load_model

from ocr.hiragana.symbols import label


class processor:
    def __init__(self):
        self.__symbol = ""
        self.__label = label

        self.__conf_thresh = 0.3
        self.__model = load_model("./model/hiragana.h5")

        self.__model.predict(np.array([np.zeros((48, 48, 1), np.float32)]))
        self.__thread = threading.Thread(
            target=self.__process_threaded,
            args=(np.zeros((48, 48, 3), np.uint8),),
        )

    def process(self, canvas):
        if not self.__thread.is_alive():
            self.__thread = threading.Thread(
                target=self.__process_threaded, args=(canvas,)
            )
            self.__thread.start()

        return self.__symbol

    def __process_threaded(self, canvas):
        symbol = ""
        blurred = cv2.GaussianBlur(canvas, (5, 5), 0)
        edged = processor.__auto_canny(blurred)
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            self.__symbol = symbol
            return
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        (x, y, w, h) = cv2.boundingRect(cnt)

        roi = canvas[y : y + h, x : x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        if tW > tH:
            thresh = imutils.resize(thresh, width=48)
        else:
            thresh = imutils.resize(thresh, height=48)

        (tH, tW) = thresh.shape
        dX = int(max(0, 48 - tW) / 2.0)
        dY = int(max(0, 48 - tH) / 2.0)

        padded = cv2.copyMakeBorder(
            thresh,
            top=dY,
            bottom=dY,
            left=dX,
            right=dX,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        padded = cv2.resize(padded, (48, 48))

        padded = padded.astype(np.float32) / 255.0
        padded = np.expand_dims(padded, axis=-1)

        char_and_bb = (padded, (x, y, w, h))

        pred = self.__model.predict(np.array([char_and_bb[0]], np.float32))[0]

        i = self.__get_pred(pred)
        if i == -1:
            self.__symbol = symbol
            return
        label = self.__label[i]
        symbol = label

        self.__symbol = symbol

    def __get_pred(self, pred):
        idx = np.argsort(pred)[::-1]

        if pred[idx[0]] < self.__conf_thresh:
            return -1

        return idx[0]

    @staticmethod
    def __auto_canny(blurred: np.array, sigma=0.33):
        v = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(blurred, lower, upper)
        return edges
