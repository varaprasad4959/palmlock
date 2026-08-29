from ultralytics import YOLO
import cv2
import torch


class HandDetector:

    def __init__(self, model_path):

        print("Loading hand detector...")

        self.model = YOLO(model_path)

        # Automatically select GPU or CPU
        if torch.cuda.is_available():
            self.device = 0
            print(
                "Hand detector device: GPU - "
                f"{torch.cuda.get_device_name(0)}"
            )
        else:
            self.device = "cpu"
            print("Hand detector device: CPU")

        print("Hand detector loaded successfully!")

    def detect(self, frame):
        """
        Detect hands in a camera frame.

        Returns:
            list of detected hands.
        """

        results = self.model(
            frame,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            if result.keypoints is None:
                continue

            for i in range(len(result.boxes)):

                confidence = float(
                    result.boxes.conf[i].cpu().item()
                )

                if confidence < 0.60:
                    continue

                box = (
                    result.boxes.xyxy[i]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                keypoints = (
                    result.keypoints.xy[i]
                    .cpu()
                    .numpy()
                )

                detections.append({
                    "confidence": confidence,
                    "box": box,
                    "keypoints": keypoints
                })

        return detections


# --------------------------------------------------
# Standalone detector test
# --------------------------------------------------

if __name__ == "__main__":

    MODEL_PATH = (
        r"C:\Users\varap\PalmLock"
        r"\models\hand_detector.pt"
    )

    detector = HandDetector(MODEL_PATH)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        raise SystemExit

    print("Camera started.")
    print("Show your hand.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read camera frame.")

            break

        # Mirror the webcam
        frame = cv2.flip(frame, 1)

        detections = detector.detect(frame)

        if detections:

            for detection in detections:

                x1, y1, x2, y2 = detection["box"]

                confidence = detection["confidence"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"HAND {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "HAND NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "PalmLock - Hand Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()