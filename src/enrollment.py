import os
import time
import cv2
import numpy as np

from detector import HandDetector
from palm import extract_palm


class PalmEnrollment:

    def __init__(self, model_path, enrollment_dir):
        self.detector = HandDetector(model_path)
        self.enrollment_dir = enrollment_dir

        os.makedirs(
            self.enrollment_dir,
            exist_ok=True
        )

    def _clear_old_samples(self):
        """Remove previous enrollment images."""

        for filename in os.listdir(self.enrollment_dir):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                path = os.path.join(
                    self.enrollment_dir,
                    filename
                )

                os.remove(path)

    def _is_different_enough(
        self,
        palm,
        previous_palm,
        threshold=8.0
    ):
        """
        Prevent capturing almost identical frames.

        Returns True if the current frame is sufficiently
        different from the previous captured frame.
        """

        if previous_palm is None:
            return True

        current = cv2.resize(
            palm,
            (64, 64)
        )

        previous = cv2.resize(
            previous_palm,
            (64, 64)
        )

        current_gray = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2GRAY
        )

        previous_gray = cv2.cvtColor(
            previous,
            cv2.COLOR_BGR2GRAY
        )

        difference = cv2.absdiff(
            current_gray,
            previous_gray
        )

        mean_difference = float(
            np.mean(difference)
        )

        return mean_difference >= threshold

    def capture_samples(
        self,
        required_samples=75,
        capture_interval=0.15
    ):
        """
        Automatically capture good palm samples.

        The user only needs to show and slightly move
        their palm. No key presses are required.
        """

        self._clear_old_samples()

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open camera."
            )

        samples = []
        previous_palm = None

        last_capture_time = 0

        print()
        print("==============================")
        print("PALM ENROLLMENT")
        print("==============================")
        print(
            f"Target samples: {required_samples}"
        )
        print()
        print("Show your palm.")
        print("Move it slowly in different directions.")
        print("Keep your palm clearly visible.")
        print("Press Q to cancel.")
        print()

        while len(samples) < required_samples:

            success, frame = camera.read()

            if not success:
                continue

            # Mirror camera view
            frame = cv2.flip(frame, 1)

            detections = self.detector.detect(frame)

            palm = None

            if detections:

                detection = detections[0]

                palm = extract_palm(
                    frame,
                    detection["keypoints"]
                )

            current_time = time.time()

            if palm is not None:

                # Show the palm being captured
                display_palm = cv2.resize(
                    palm,
                    (300, 300)
                )

                cv2.imshow(
                    "Palm Preview",
                    display_palm
                )

                enough_time_passed = (
                    current_time - last_capture_time
                    >= capture_interval
                )

                different_frame = (
                    self._is_different_enough(
                        palm,
                        previous_palm
                    )
                )

                if (
                    enough_time_passed
                    and different_frame
                ):

                    filename = (
                        f"palm_{len(samples):03d}.jpg"
                    )

                    path = os.path.join(
                        self.enrollment_dir,
                        filename
                    )

                    cv2.imwrite(
                        path,
                        palm
                    )

                    samples.append(path)

                    previous_palm = palm.copy()

                    last_capture_time = current_time

            # Main camera display
            cv2.putText(
                frame,
                f"Palm samples: "
                f"{len(samples)}/{required_samples}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            if palm is not None:

                cv2.putText(
                    frame,
                    "PALM DETECTED - MOVE SLOWLY",
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "SHOW YOUR PALM",
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "PalmLock - Enrollment",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

        print()
        print("==============================")
        print("ENROLLMENT COMPLETE")
        print("==============================")
        print(
            f"Samples captured: "
            f"{len(samples)}"
        )

        return samples


if __name__ == "__main__":

    MODEL_PATH = (
        r"C:\Users\varap\PalmLock"
        r"\models\hand_detector.pt"
    )

    ENROLLMENT_DIR = (
        r"C:\Users\varap\PalmLock"
        r"\data\enrollment"
    )

    enrollment = PalmEnrollment(
        MODEL_PATH,
        ENROLLMENT_DIR
    )

    enrollment.capture_samples(
        required_samples=75
    )