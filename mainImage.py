import cv2

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Configure path to project
default_path = "./"

cfg_path = default_path + "/nn.cfg"
weights_path = default_path + "/nn_best.weights"

class_names = []
with open("nn.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Put image path here
image_path = "./imgAndVideo/img1.jpg"
frame = cv2.imread(image_path)

net = cv2.dnn.readNet(cfg_path, weights_path)
model = cv2.dnn_DetectionModel(net)

# Configure network dimensions according to cfg file
model.setInputParams(size=(608, 608), scale=1/255) 

classes, scores, boxes = model.detect(frame, 0.1, 0.2)

for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = f"{class_names[classid]} : {score:.2f}"
    cv2.rectangle(frame, box, color, 2)
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Hit "esc" to leave window, or bugs will occur
cv2.imshow("Detections", frame)
cv2.waitKey(0) 
cv2.destroyAllWindows()
