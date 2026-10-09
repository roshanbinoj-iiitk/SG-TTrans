---
name: computer-vision
description: >-
  Use this skill when developing computer vision pipelines, image processing workflows with OpenCV (cv2),
  bounding box manipulations, torchvision/Albumentations augmentations, classical CV algorithms,
  video streaming, and converting between OpenCV arrays and PyTorch vision tensors.
---

# Computer Vision & OpenCV Engineering Guidelines

This skill provides patterns, workflows, and best practices for building computer vision systems using OpenCV, NumPy, and PyTorch.

---

## 1. OpenCV Core Conventions

OpenCV follows distinct conventions that differ from Matplotlib, PIL, and PyTorch:

| Concept | OpenCV Convention | Matplotlib / PIL / PyTorch |
| :--- | :--- | :--- |
| **Color Order** | **BGR** (`Blue, Green, Red`) | **RGB** (`Red, Green, Blue`) |
| **Array Indexing** | `image[row, col]` = `image[y, x]` | `(y, x)` in array slices |
| **Point Coordinates** | `cv2.circle(img, (x, y), ...)` | `(x, y)` Cartesian coordinates |
| **Tensor Layout** | Channels Last: `(H, W, C)` | PyTorch Channels First: `(C, H, W)` or `(B, C, H, W)` |

### Safe Loading & Color Conversions
```python
from pathlib import Path
import cv2
import numpy as np

def load_rgb_image(path: Path | str) -> np.ndarray:
    """Loads image from disk and converts from OpenCV BGR to RGB."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_rgb_image(path: Path | str, rgb_image: np.ndarray) -> None:
    """Converts RGB image back to BGR before writing with OpenCV."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
```

---

## 2. Image Resizing & Geometric Warping

Always choose the correct interpolation algorithm based on the transformation:
* **Downsampling (Shrinking)**: Use `cv2.INTER_AREA` (avoids aliasing artifacts).
* **Upsampling (Enlarging)**: Use `cv2.INTER_CUBIC` (higher quality) or `cv2.INTER_LINEAR` (faster).
* **Segmentation Masks / Labels**: Always use `cv2.INTER_NEAREST` to preserve discrete class IDs.

```python
def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize image to (width, height) using appropriate interpolation."""
    target_w, target_h = target_size
    orig_h, orig_w = image.shape[:2]

    if target_w < orig_w and target_h < orig_h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
```

---

## 3. Bounding Box Transformations

Standardize coordinate formats across your pipeline:

```python
import numpy as np

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]."""
    xyxy = boxes.copy()
    xyxy[..., 2] = boxes[..., 0] + boxes[..., 2]
    xyxy[..., 3] = boxes[..., 1] + boxes[..., 3]
    return xyxy

def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x_min, y_min, x_max, y_max] to [center_x, center_y, width, height]."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=-1)

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
```

---

## 4. Bridge Between OpenCV and PyTorch

Efficient conversion between OpenCV NumPy arrays and PyTorch vision tensors without unnecessary memory copies:

```python
import numpy as np
import torch

def cv2_to_torch_tensor(rgb_img: np.ndarray) -> torch.Tensor:
    """
    Converts HxWxC uint8 RGB NumPy array to 1xCxHxW float32 PyTorch tensor in [0.0, 1.0].
    """
    # 1. HxWxC -> CxHxW
    tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).contiguous()
    # 2. Cast to float32 and normalize [0, 1]
    tensor = tensor.float() / 255.0
    # 3. Add batch dimension: 1xCxHxW
    return tensor.unsqueeze(0)

def torch_tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts 1xCxHxW or CxHxW float32 PyTorch tensor in [0.0, 1.0] back to HxWxC uint8 RGB array.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np
```

---

## 5. Classical Vision Operations & Contours

```python
import cv2
import numpy as np

def extract_contours(bgr_img: np.ndarray, min_area: float = 100.0) -> list[np.ndarray]:
    """Finds external contours with Otsu thresholding and morphological cleanup."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill minor holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
```

---

## 6. Real-Time Video Streaming Loop

Robust frame-by-frame loop with FPS tracking and resource safety:

```python
from pathlib import Path
import cv2

def process_video_stream(source: int | str | Path) -> None:
    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream at {source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame processing logic here
            # ...

    finally:
        cap.release()
        cv2.destroyAllWindows()
```
