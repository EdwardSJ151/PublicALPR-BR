import cv2
import time

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Configure path to project
default_path = "./"

cfg_path = default_path + "/nn.cfg"
weights_path = default_path + "/nn_best.weights"

class_names = []
# with open("coco.names", "r") as f:
with open("nn.names", "r") as f:  
    class_names = [cname.strip() for cname in f.readlines()]

# 0 is for webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./inferencia/video.mp4")

net = cv2.dnn.readNet(cfg_path, weights_path)

# Input model size acording to cfg file
model = cv2.dnn_DetectionModel(net)
# model.setInputParams(size=(416, 416), scale=1/255) #swapRB=True
# model.setInputParams(size=(512, 288), scale=1/255) #swapRB=True
model.setInputParams(size=(608, 608), scale=1/255) #swapRB=True

# Frame reading
while True:
    _, frame = cap.read()
    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end = time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid]} : {score}" # label = f"{class_names[classid[0]} : {score}"

        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("detections", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()