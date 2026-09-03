import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGE_SIZE = 32

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BACKEND_DIR / "char_classifier.pt"
DEFAULT_LABELS_PATH = BACKEND_DIR / "char_classifier_labels.json"

# Row-clustering and character-blob filters, tuned against real plate
# crops (see scripts/train_char_classifier.py for how the model itself
# was trained). Kept here since segmentation and classification are used
# together everywhere.
MIN_CHAR_AREA_FRAC = 0.005
MAX_CHAR_AREA_FRAC = 0.5
MIN_CHAR_HEIGHT_FRAC = 0.15
ROW_Y_TOLERANCE_FRAC = 0.6
CROP_MARGIN_FRAC = 0.12

# Not a character -- a logo/watermark class present on embossed private-
# vehicle plates, included in training so the classifier has somewhere
# to put it instead of forcing it into a real character class, but
# dropped from the assembled text.
NON_TEXT_LABELS = {"Nepali Flag"}


class CharClassifierCNN(nn.Module):
    """34-class CNN over 32x32 grayscale crops of individual Nepali plate
    characters (Devanagari digits + the province/category syllables that
    appear on plates). See scripts/train_char_classifier.py for training.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def segment_characters(plate_crop):
    """Finds individual character blobs in a plate crop via classical CV
    (Otsu threshold + connected components) and orders them into
    top-to-bottom rows, left-to-right within each row.

    Nepali plates are high-contrast by design (stamped/embossed
    characters in one color on a solid background), which is exactly
    where Otsu + connected-components segmentation works well -- no
    detector training needed for this step, unlike character
    *identification*, which does need a trained classifier (see
    CharClassifierCNN) because the polarity/color varies by plate
    category (red government plates, white/black private plates, etc.)
    while the character shapes stay the model's problem, not
    segmentation's.

    Returns a list of rows, each row a list of (x, y, w, h) boxes in
    plate_crop's coordinate space, ordered for left-to-right, top-to-
    bottom reading.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Otsu just splits pixels into two classes; figure out which one is
    # the text. Character strokes cover a minority of the plate's area,
    # so the smaller class is the foreground -- this works regardless of
    # whether a given plate has light text on dark background or the
    # reverse.
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_frac = (w * h) / (width * height)

        if area_frac < MIN_CHAR_AREA_FRAC or area_frac > MAX_CHAR_AREA_FRAC:
            continue

        if h < height * MIN_CHAR_HEIGHT_FRAC:
            continue

        boxes.append((x, y, w, h))

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[1])
    rows = []

    for box in boxes:
        _, y, _, h = box
        center_y = y + h / 2
        placed = False

        for row in rows:
            row_center_y = sum(b[1] + b[3] / 2 for b in row) / len(row)
            row_height = sum(b[3] for b in row) / len(row)

            if abs(center_y - row_center_y) < row_height * ROW_Y_TOLERANCE_FRAC:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    rows.sort(key=lambda row: sum(b[1] for b in row) / len(row))

    return [sorted(row, key=lambda b: b[0]) for row in rows]


class CharClassifier:

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        labels_path=DEFAULT_LABELS_PATH
    ):
        self.labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.model = CharClassifierCNN(len(self.labels))
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        self.model.eval()

    def read_plate_text(self, plate_crop):
        """Segments plate_crop into characters and classifies each.
        Returns (text, mean_confidence), or (None, None) if fewer than
        2 characters were segmented -- too little signal to trust over
        the EasyOCR fallback.
        """
        rows = segment_characters(plate_crop)
        char_count = sum(len(row) for row in rows)

        if char_count < 2:
            return None, None

        height, width = plate_crop.shape[:2]
        crops = []
        row_lengths = []

        for row in rows:
            row_lengths.append(len(row))

            for (x, y, w, h) in row:
                margin_x = int(w * CROP_MARGIN_FRAC)
                margin_y = int(h * CROP_MARGIN_FRAC)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(width, x + w + margin_x)
                y2 = min(height, y + h + margin_y)
                crops.append(plate_crop[y1:y2, x1:x2])

        labels, confidences = self._classify_batch(crops)

        rows_text = []
        text_confidences = []
        idx = 0

        for length in row_lengths:
            row_chars = []

            for label, confidence in zip(
                labels[idx:idx + length], confidences[idx:idx + length]
            ):
                if label in NON_TEXT_LABELS:
                    continue

                row_chars.append(label)
                text_confidences.append(confidence)

            rows_text.append("".join(row_chars))
            idx += length

        if not text_confidences:
            return None, None

        text = " ".join(rows_text)
        mean_confidence = round(float(np.mean(text_confidences)), 4)

        return text, mean_confidence

    def _classify_batch(self, crops):
        tensors = [self._preprocess(crop) for crop in crops]
        batch = torch.stack(tensors)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            confidences, indices = probs.max(dim=1)

        labels = [self.labels[i] for i in indices.tolist()]

        return labels, confidences.tolist()

    @staticmethod
    def _preprocess(crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )
        return torch.from_numpy(resized).float().unsqueeze(0) / 255.0
