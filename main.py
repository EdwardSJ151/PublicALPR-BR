import cv2
import os
import time
import argparse

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


def load_model(model_path, network_size):
    """Load the model and class names."""
    cfg_path = os.path.join(model_path, "nn.cfg")
    weights_path = os.path.join(model_path, "nn_best.weights")
    names_path = os.path.join(model_path, "nn.names")

    # Load the class names
    with open(names_path, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    # Load the network
    net = cv2.dnn.readNet(cfg_path, weights_path)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=network_size, scale=1 / 255)

    return model, class_names


def process_image(image_path, model, class_names, output_folder, model_path):
    """Perform inference on an image."""
    frame = cv2.imread(image_path)
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid]} : {score:.2f}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save result
    os.makedirs(output_folder, exist_ok=True)
    last_folder = os.path.basename(os.path.normpath(model_path))
    output_file = f"{os.path.splitext(os.path.basename(image_path))[0]}_{last_folder}_result.jpg"
    output_path = os.path.join(output_folder, output_file)
    cv2.imwrite(output_path, frame)
    print(f"Image result saved to {output_path}")


def process_video(video_path, model, class_names, output_folder, model_path):
    """Perform inference on a video."""
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    last_folder = os.path.basename(os.path.normpath(model_path))
    output_file = f"{os.path.splitext(os.path.basename(video_path))[0]}_{last_folder}_result.avi"
    output_path = os.path.join(output_folder, output_file)


    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        classes, scores, boxes = model.detect(frame, 0.1, 0.2)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = f"{class_names[classid]} : {score:.2f}"
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video result saved to {output_path}")


def process_webcam(model, class_names):
    """Perform inference on webcam feed."""
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        start = time.time()
        classes, scores, boxes = model.detect(frame, 0.1, 0.2)
        end = time.time()

        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = f"{class_names[classid]} : {score:.2f}"
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fps_label = f"FPS: {round(1.0 / (end - start), 2)}"
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Detections", frame)

        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run inference on an image, video, or webcam feed.")
    parser.add_argument("--mode", type=str, choices=["image", "video", "webcam"], required=True, help="Inference mode: 'image', 'video', or 'webcam'")
    parser.add_argument("--model_path", type=str, default="./model/mercosulFullPlate", help="Path to the model directory")
    parser.add_argument("--input_path", type=str, help="Path to the input image or video file (required for image/video modes)")
    parser.add_argument("--output_path", type=str, default="./inferenceResults", help="Path to save the inference results (for image/video modes)")
    parser.add_argument("--network_size", type=str, default="992,736", help="Network input size as 'width,height' (default: 992,736)")
    args = parser.parse_args()

    # Parse network size
    network_size = tuple(map(int, args.network_size.split(',')))

    # Load the model
    model, class_names = load_model(args.model_path, network_size)

    if args.mode == "image":
        if not args.input_path:
            raise ValueError("Input path is required for image mode")
        process_image(args.input_path, model, class_names, args.output_path, args.model_path)

    elif args.mode == "video":
        if not args.input_path:
            raise ValueError("Input path is required for video mode")
        process_video(args.input_path, model, class_names, args.output_path, args.model_path)

    elif args.mode == "webcam":
        process_webcam(model, class_names)


if __name__ == "__main__":
    main()
