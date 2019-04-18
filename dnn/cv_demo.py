import cv2 as cv
import numpy as np

cvNet = cv.dnn.readNetFromTensorflow('/home/mars/Mobilenet_v2/cnkrflags/v2_cn_graph.pb', '/home/mars/Mobilenet_v2/cnkrflags/v2_cn_graph.pbtxt')

#cvNet = cv.dnn.readNetFromTensorflow('/home/mars/sorted_inference_graph.pb', '/home/mars/v1_ssd.pbtxt')
#img = cv.imread('/home/mars/air/image32.jpg')
img = cv.imread('/home/mars/7.jpg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, 1.0,size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', img)
#cv.imwrite("test7.jpg",img)
cv.waitKey()
