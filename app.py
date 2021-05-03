import argparse
from typing import Dict, List

import albumentations as A
import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from facenet_pytorch import MTCNN

from utils import create_classification_model, create_segmentation_model
from utils import haze_removal, fill_and_close


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=str, help="Path to the config.", required=True)
    return parser.parse_args()


class TongueAnalysisApp(object):
    def __init__(self, segmentation_model, classification_model, face_detector,
                 class_names: List[str] = None,
                 segmentation_threshold: float = 0.3,
                 class_count: int = 5,
                 rules: Dict = None):

        self.segmentation_model = segmentation_model
        self.classification_model = classification_model
        self.face_detector = face_detector

        self.segmentation_preprocess = A.Compose([
            A.LongestMaxSize(max_size=512, p=1.0),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        ], p=1.0)
        self.segmentation_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)

        self.denorm = A.Normalize(mean=(-0.485 * 0.229, -0.456 * 0.224, -0.406 * 0.255),
                                  std=(1 / 0.229, 1 / 0.224, 1 / 0.255), p=1.0)

        self.classification_preprocess = A.Compose([
            A.LongestMaxSize(max_size=512, p=1.0),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)
        ], p=1.0)

        self.threshold = segmentation_threshold

        self.class_count = class_count

        if class_names is None:
            self.class_names = [str(i) for i in range(self.class_count)]
        else:
            self.class_names = class_names

        self.rules = rules

        self.iface = None

    def setup(self):
        self.segmentation_model.eval()
        self.classification_model.eval()
        self.face_detector.eval()

        self.iface = gr.Interface(self.procees,
                                  gr.inputs.Image(shape=(2048, 2048),
                                                  source=self.rules["input_type"],
                                                  tool=None),
                                  gr.outputs.Label(num_top_classes=self.class_count),
                                  "image")
        self.iface.test_launch()

    def launch(self):
        self.iface.launch(share=True)

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.rules["use_segmentation"]:
            print("Segmentation")

            preprocessed = self.segmentation_preprocess(image=image)["image"]

            normalized = self.segmentation_norm(image=preprocessed)["image"]

            tensor_img = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze_(0).cuda()

            predictions = self.segmentation_model(tensor_img)

            segmentation_result = (predictions[0][0].cpu().numpy() > self.threshold).astype(np.uint8) * 255

            segmentation_result = fill_and_close(segmentation_result)

            segmentation_result = cv2.cvtColor(segmentation_result, cv2.COLOR_GRAY2BGR)

            # Projection
            result = cv2.bitwise_and(preprocessed, segmentation_result)

            return result
        return image

    @torch.no_grad()
    def classify(self, image: np.ndarray) -> Dict[str, float]:
        print("Classification")
        normalized = self.classification_preprocess(image=image)["image"]
        tensor_img = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze_(0).cuda()

        predictions = self.classification_model(tensor_img)[0].cpu().numpy()

        return {self.class_names[i]: float(predictions[i]) for i in range(self.class_count)}

    @torch.no_grad()
    def detect(self, image: np.ndarray, bottom_margin: int = 100) -> np.ndarray:
        print("Face Detection")
        # tensor_img = torch.from_numpy(image)
        boxes, probs = self.face_detector.detect(image)

        if boxes is None:
            return image

        x1, y1, x2, y2 = boxes[0]

        result = image[int(y1): min(int(y2) + bottom_margin, image.shape[0]), int(x1): int(x2)]

        return result

    def procees(self, image: np.ndarray) -> Dict[str, float]:
        image = self.detect(image, bottom_margin=70)

        # image = haze_removal(image)

        cv2.imwrite("face.png", image)
        cv2.imwrite("face_bgr.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        h, w = image.shape[:2]

        # Get the tongue region
        image = image[int(h * 0.65): h, w // 8: w - w // 8]

        cv2.imwrite("tongue.png", image)

        segmentation_result = self.segment(image)

        cv2.imwrite("segmentation.png", segmentation_result)

        classification_result = self.classify(segmentation_result)

        return classification_result


def main():
    args = get_args()

    with open(args.config_path) as fp:
        h_params = yaml.load(fp, Loader=yaml.SafeLoader)

    classification_model = create_classification_model(h_params=h_params["classification_model"],
                                                       weight_path=h_params["classification_model_weights"])
    segmentation_model = create_segmentation_model(h_params=h_params["segmentation_model"],
                                                   weight_path=h_params["segmentation_model_weights"])

    face_detector = MTCNN(image_size=500)

    app = TongueAnalysisApp(segmentation_model=segmentation_model,
                            classification_model=classification_model,
                            face_detector=face_detector,
                            class_names=sorted(h_params["class_names"]),
                            segmentation_threshold=0.7,
                            class_count=h_params["classification_model"]["classes"],
                            rules=h_params["rules"])
    app.setup()
    app.launch()


if __name__ == "__main__":
    main()
