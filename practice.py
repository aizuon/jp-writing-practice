import cv2
import numpy as np

# from numba import config

# config.THREADING_LAYER = "tbb"

from utils import pt_in_line_w_tol
from ocr.processor import processor


canvas = np.full((480, 480, 1), 255, np.uint8)

drawing = False
erasing = False

lines = []
last_pt = None


def draw(event, x, y, flags, param):
    global drawing, erasing, canvas, last_pt, lines
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_pt = None
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    if event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False
    if drawing:
        if last_pt:
            lines.append((last_pt, (x, y)))
        last_pt = (x, y)

        canvas[:] = 255
        for line in lines:
            cv2.line(canvas, line[0], line[1], 0, 5, cv2.LINE_AA)
    if erasing:
        idx_to_del = []
        for i in range(len(lines)):
            if pt_in_line_w_tol(lines[i][0], lines[i][1], (x, y)):
                idx_to_del.append(i)
        for i in sorted(idx_to_del, reverse=True):
            del lines[i]

        canvas[:] = 255
        for line in lines:
            cv2.line(canvas, line[0], line[1], 0, 5, cv2.LINE_AA)


window_name = "canvas"
ocr_processor = processor()
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, draw)
while True:
    cv2.imshow(window_name, canvas)
    symbol = ocr_processor.process(canvas)
    if symbol:
        print(symbol)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
