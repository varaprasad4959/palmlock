import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms


class PalmRecognizer:

    def __init__(self, database_path):

        print("Loading palm recognition model...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Device:", self.device)

        if self.device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        self.model = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=0
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.database = np.load(database_path)

        print(
            "Database shape:",
            self.database.shape
        )

        print("Palm recognition model ready!")

    def extract_embedding(self, palm_image):

        if palm_image is None:
            return None

        # OpenCV BGR → RGB
        rgb_image = cv2_to_rgb(palm_image)

        image = Image.fromarray(rgb_image)

        tensor = self.transform(
            image
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            embedding = self.model(tensor)

            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1
            )

        return embedding.squeeze(0).cpu().numpy()


def cv2_to_rgb(image):
    """
    Convert an OpenCV BGR image to RGB.
    """

    return image[:, :, ::-1].copy()