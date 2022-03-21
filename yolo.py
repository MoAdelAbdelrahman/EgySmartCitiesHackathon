import cv2
import numpy as np
import argparse
import time
Prediction_Threshold = 0.5
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
args = parser.parse_args()


def readLabels():
    f = open('obj.names').read().strip().split('\n')
    return f


def loadNetwork():
    mainNet = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    mainNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    mainNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    classes = readLabels()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    layersNames = mainNet.getLayerNames()
    output_layers = [layersNames[i - 1] for i in mainNet.getUnconnectedOutLayers()]
    return mainNet, classes, colors, output_layers


def loadImage(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def openCam():
    mainCam = cv2.VideoCapture(0)
    return mainCam


def detection(img, network, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392,size=(416,416), mean=(0,0, 0), swapRB=False, crop=False)
    network.setInput(blob)
    outputs = network.forward(output_layers)
    return blob, outputs


def createBoxes(outputs, h, w):
    boxes = []
    confidences = []
    classIDs = []

    for out in outputs:
        for obj in out:
            scores = obj[5:]  # class probabilities
            class_id = np.argmax(scores)  # max score is what I want
            conf = scores[class_id]
            if conf > Prediction_Threshold:
                centerX = int(obj[0] * w)
                centerY = int(obj[1] * h)
                box_w = int(obj[2] * w)
                box_h = int(obj[3] * h)
                x = int(centerX - (box_w / 2))
                y = int(centerY - (box_h / 2))
                box = [x, y, box_w, box_h]
                boxes.append(box)
                confidences.append(conf)
                classIDs.append(class_id)
    return boxes, confidences, classIDs


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    g = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confID = str(confs[i])
            #print(i)
            color = colors[g]
            g += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            cv2.putText(img, confID, (x + 50, y - 5), font, 1, color, 1)
    img = cv2.resize(img, (800, 600))
    cv2.imshow("Image", img)


def image_detect(img_path):
    model, classes, colors, output_layers = loadNetwork()
    image, height, width, channels = loadImage(img_path)
    blob, outputs = detection(image, model, output_layers)
    boxes, confs, class_ids = createBoxes(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

def webcam_detect():
    model, classes, colors, output_layers = loadNetwork()
    cap = openCam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detection(frame, model, output_layers)
        boxes, confs, class_ids = createBoxes(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


if __name__ == '__main__':
    webcam = args.webcam
    print('---- Starting Web Cam object detection ----')
    webcam_detect()

    cv2.destroyAllWindows()



'''
cam = openCam()
video = cam.read()
time.sleep(3)
cv2.imshow('Detection', video)
cv2.waitKey(27)
video.release()
'''

