"""
Trains a small CNN to classify individual Nepali plate characters.

Why this exists: EasyOCR's generic 'ne' Devanagari model is trained on
printed/handwritten documents, not the blocky embossed stencil font used
on Nepali plates — it reads garbage on real plate crops regardless of
preprocessing (upscaling, CLAHE, etc. were all tried and don't help,
since the problem is a font/domain mismatch, not resolution). This
trains a classifier on real plate character crops instead.

Three sources, merged into one 64-class problem:

1. Kaggle "Nepali Number Plate Characters Dataset"
   (inspiring-lab/nepali-number-plate-characters-dataset, CC BY-NC 4.0 --
   non-commercial use only). 26,537 real 32x32 crops across 34 classes:
   Devanagari digits 0-9 plus the specific province/vehicle-category
   syllables that actually appear on government-style plates (e.g.
   "प्र", "बा", "को").

       kaggle datasets download -d inspiring-lab/nepali-number-plate-characters-dataset \\
           -p backend/datasets/nepali_char_ocr --unzip

2. Kaggle "Embossed Number Plate Character Dataset (Nepal V)"
   (theciceerguy/embossed-number-plate-detection-nepal-dataset, MIT).
   ~3,500 crops across 29 classes: 0-9 and A-Z (minus letters skipped to
   avoid digit lookalikes) plus a "Nepali Flag" logo/watermark class --
   the newer embossed Latin-script plates used on private vehicles.

       kaggle datasets download -d theciceerguy/embossed-number-plate-detection-nepal-dataset \\
           -p backend/datasets/embossed_chars --unzip

   Without this second dataset, the classifier only ever knows
   Devanagari classes -- on a Latin-script plate it would still confidently
   (and wrongly) force every character into the closest-looking
   Devanagari class instead of recognizing it doesn't know this script.
   Verified: a synthetic Latin plate crop was misread as Devanagari
   garbage at 0.76 confidence (above the fallback threshold) before this
   was added.

3. A "background" class of generic vehicle-body texture (grilles,
   windshields, seats, fabric, etc.) -- NOT downloaded, harvested
   locally from real photos via harvest_char_negatives.py. Without
   this, the classifier has no way to say "this segmented blob isn't a
   character at all" -- verified this was a real problem: testing
   PlateRecognizer against real multi-vehicle Nepal traffic photos
   showed the plate *detector* proposing false-positive boxes on busy
   scenes (a van grille, a windshield, a motorcycle seat, fabric), and
   the classifier confidently reading plausible-looking Devanagari
   garbage from every one of them instead of recognizing they weren't
   plates. See harvest_char_negatives.py's docstring for the harvesting
   approach; it writes to backend/datasets/char_negatives/.

All three are merged into backend/datasets/combined_char_ocr/ (one
class-name subfolder per character, images symlinked in) before
training:

    SRC_DEV=backend/datasets/nepali_char_ocr/character_ocr
    SRC_LAT=backend/datasets/embossed_chars/digits_and_numbers_dataset
    SRC_NEG=backend/datasets/char_negatives
    DST=backend/datasets/combined_char_ocr
    for d in "$SRC_DEV"/*/; do
      cls=$(basename "$d"); mkdir -p "$DST/$cls"
      for f in "$d"*; do ln -s "$f" "$DST/$cls/$(basename "$f")"; done
    done
    for split in train valid test; do
      for d in "$SRC_LAT/$split"/*/; do
        cls=$(basename "$d"); mkdir -p "$DST/$cls"
        for f in "$d"*; do ln -s "$f" "$DST/$cls/${split}_$(basename "$f")"; done
      done
    done
    mkdir -p "$DST/background"
    for f in "$SRC_NEG"/*; do ln -s "$f" "$DST/background/$(basename "$f")"; done

Usage:
    python backend/scripts/train_char_classifier.py

Trains for 15 epochs on CPU (~30k images, 32x32 grayscale -- a few
minutes total), then saves weights to
backend/char_classifier.pt and the label mapping to
backend/char_classifier_labels.json. Both are required by
PlateRecognizer's character-segmentation OCR path.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.char_classifier import CharClassifierCNN  # noqa: E402

DATA_DIR = BACKEND_DIR / "datasets" / "combined_char_ocr"
MODEL_OUT = BACKEND_DIR / "char_classifier.pt"
LABELS_OUT = BACKEND_DIR / "char_classifier_labels.json"

IMAGE_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 15
VAL_FRACTION = 0.15


class _TransformedSubset(torch.utils.data.Dataset):
    """Wraps a Subset with its own transform (train needs augmentation,
    val doesn't) since random_split shares the underlying ImageFolder.
    Defined at module level so DataLoader worker processes can pickle it.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Dataset not found at {DATA_DIR}. Download it first (see "
            "this script's docstring)."
        )

    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(str(DATA_DIR))
    class_names = full_dataset.classes

    val_size = int(len(full_dataset) * VAL_FRACTION)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        _TransformedSubset(train_subset, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        _TransformedSubset(val_subset, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {train_size}, Val: {val_size}")

    model = CharClassifierCNN(len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=6, gamma=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= train_size
        scheduler.step()

        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = correct / val_size
        print(
            f"epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)

    LABELS_OUT.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best val_acc={best_val_acc:.4f}. Saved {MODEL_OUT}, {LABELS_OUT}")


if __name__ == "__main__":
    main()
