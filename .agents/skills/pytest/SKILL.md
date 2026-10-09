---
name: pytest
description: >-
  Use this skill when writing, structuring, or running Python tests with pytest, creating fixtures in conftest.py,
  parameterizing tests, mocking dependencies, and validating PyTorch models, tensor shapes, and computer vision pipelines.
---

# Pytest & Python Testing Guidelines

This skill provides patterns, workflows, and best practices for writing clean, fast, and comprehensive automated test suites using `pytest`.

---

## 1. Test Suite Architecture

Standard layout for Python projects:

```text
my_project/
├── pyproject.toml
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── models.py
│       └── utils.py
└── tests/
    ├── conftest.py          # Shared fixtures across tests
    ├── unit/
    │   ├── test_models.py
    │   └── test_utils.py
    └── integration/
        └── test_pipeline.py
```

* **Naming Conventions**:
  - Test files: `test_*.py` or `*_test.py`
  - Test classes: `Test*` (no `__init__`)
  - Test functions: `test_*`

---

## 2. Fixtures & `conftest.py`

Use fixtures to manage reusable state, mock environments, and tear down resources cleanly.

```python
# tests/conftest.py
import numpy as np
import pytest
import torch

@pytest.fixture(scope="session")
def seed_session() -> None:
    """Ensures deterministic random states across tests."""
    torch.manual_seed(42)
    np.random.seed(42)

@pytest.fixture
def dummy_rgb_image() -> np.ndarray:
    """Provides a synthetic 256x256x3 RGB uint8 image fixture."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

@pytest.fixture
def dummy_batch_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Provides a batch of images (B, C, H, W) and labels."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))
    return images, labels
```

### Resource Cleanup with `yield`
```python
@pytest.fixture
def temp_checkpoint(tmp_path):
    ckpt_file = tmp_path / "model.pt"
    yield ckpt_file
    if ckpt_file.exists():
        ckpt_file.unlink()
```

---

## 3. Parameterized Testing (`@pytest.mark.parametrize`)

Test multiple inputs and edge cases with a single test function:

```python
import pytest
import torch

@pytest.mark.parametrize("in_channels,out_channels", [
    (1, 16),
    (3, 32),
    (4, 64),
])
def test_conv_layer_channels(in_channels: int, out_channels: int):
    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3)
    x = torch.randn(2, in_channels, 64, 64)
    out = layer(x)
    assert out.shape == (2, out_channels, 62, 62)
```

---

## 4. PyTorch & Computer Vision Specific Testing Patterns

### A. Model Forward Pass Smoke Test
Verify that the model processes valid shapes without crashing and produces expected output shapes:

```python
def test_model_forward_shape(dummy_batch_tensors):
    images, _ = dummy_batch_tensors
    model = MyVisionClassifier(num_classes=10)
    model.eval()

    with torch.inference_mode():
        output = model(images)

    assert output.shape == (4, 10)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
```

### B. Gradient Flow & Backward Pass Check
Verify that all learnable parameters receive gradients during backpropagation:

```python
def test_model_gradient_flow(dummy_batch_tensors):
    images, labels = dummy_batch_tensors
    model = MyVisionClassifier(num_classes=10)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
```

### C. Numerical & Tensor Assertions
Use `torch.allclose` or `np.testing.assert_allclose` with tolerances instead of strict equality `==` for floating point comparisons:

```python
import numpy as np

def test_box_iou_computation():
    box_a = np.array([0, 0, 10, 10], dtype=float)
    box_b = np.array([0, 0, 10, 10], dtype=float)
    iou = compute_iou(box_a, box_b)
    np.testing.assert_allclose(iou, 1.0, atol=1e-6)
```

---

## 5. Mocking External Dependencies

Use `unittest.mock` or `pytest-mock` (`mocker`) to isolate units from external APIs, camera hardware, or file downloads:

```python
from unittest.mock import MagicMock, patch

def test_video_capture_error_handling():
    with patch("cv2.VideoCapture") as mock_cap:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance

        with pytest.raises(RuntimeError, match="Could not open video stream"):
            process_video_stream("camera.mp4")
```

---

## 6. Execution Commands & Useful Flags

Run tests efficiently during development:

| Command | Purpose |
| :--- | :--- |
| `pytest` | Run entire test suite |
| `pytest -v` | Verbose output displaying each test name |
| `pytest -s` | Do not capture stdout (display `print` statements) |
| `pytest -k "pattern"` | Run tests matching keyword expression |
| `pytest -x` | Stop immediately upon the first failure |
| `pytest --lf` | Run only the tests that failed in the last run |
| `pytest --cov=src --cov-report=term-missing` | Run with test coverage report |
