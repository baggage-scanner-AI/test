import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

object_names = []

with open('coco_names.txt', 'r') as f:
    # read all objects from coco_names file
    # write all object to list of object_names
    object_names = [object for object in f.read().split('\n')]


config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight_path = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weight_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:

    success, img = cap.read()
    class_ids, confs, bbox = net.detect(img, confThreshold = 0.5)
    print(class_ids, bbox)    

    if len(class_ids) != 0:    
        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            
            cv2.rectangle(img, box, color=(0, 0, 255), thickness = 2)
            cv2.putText(
                img,
                object_names[class_id - 1],
                (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                2)

    cv2.imshow('Output', img)

    cv2.waitKey(1)

