import torch
from ultralytics import YOLO

class YOLODetector:
    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str, device: str | None = None):
        self.device = self._resolve_device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect(self, frame):
        """
        Returns detections in DeepSORT format:
        [ [x, y, w, h], confidence, class_id ]
        Only PERSON detections are returned.
        """
        result = self.model(
            frame,
            device=self.device,
            classes=[self.PERSON_CLASS_ID],
            verbose=False
        )[0]

        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id != self.PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            detections.append(
                ([float(x1), float(y1), float(w), float(h)],
                 float(box.conf),
                 cls_id)
            )
        return detections
