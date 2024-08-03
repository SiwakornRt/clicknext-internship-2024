from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv

# Load YOLO model
model = YOLO("yolov8n.pt")


def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)

    for box in boxes:
        class_id = box.cls
        # print(f'Class ID : {class_id}\n')
        class_name = model.names[int(class_id)]
        # print(f'Class Name : {class_name}\n')
        coordinator = box.xyxy[0]
        # print(f'Coordinator : {coordinator}\n')
        confidence = box.conf
        # print(f'Confidence : {confidence}')

    # Draw bounding box
    annotator.box_label(
        box=coordinator, label=class_name, color=(255, 0, 0)
    )

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect only cat from class 15
    cat_class = 15

    # Detect object from image frame
    results = model.predict(frame, classes=cat_class, conf=0.95)
    # print(f'---- results {results} ----')
    for result in results:
        # print(f'---- result {result.boxes} ----')
        if result.boxes.shape[0] != 0:
            frame = draw_boxes(frame, result.boxes)
        
    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    video_writer = cv.VideoWriter(
        video_path + "_demo.avi", cv.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)
    )

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect cat from image frame
            frame_result = detect_object(frame)

            cv.putText(frame_result, 'Siwakorn-Clicknext-Internship-2024', (670, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

            # Write result to video
            video_writer.write(frame_result)

            # Show result
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", frame_result)
            cv.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    video_writer.release()
    cap.release()
    cv.destroyAllWindows()
