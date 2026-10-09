---
name: modern-python
description: >-
  Use this skill when writing, refactoring, or reviewing Python code to ensure modern Python standards
  (Python 3.10–3.12+), proper type annotations, structured data modeling (dataclasses/Pydantic v2),
  clean project management with pyproject.toml and uv, and strict linting/formatting with ruff.
---

# Modern Python Development Guidelines

This skill defines the technical standards, idioms, and tooling for developing clean, robust, and maintainable Python code targeting Python 3.10+.

---

## 1. Type Annotations & Type Safety

Always use modern, standard-library typing features. Avoid obsolete imports from `typing` where built-in alternatives exist.

### Modern Typing Rules
* **Union Types**: Use the pipe operator `X | Y` instead of `Union[X, Y]`.
* **Optional Types**: Use `X | None` instead of `Optional[X]`.
* **Built-in Generics**: Use `list[T]`, `dict[K, V]`, `set[T]`, `tuple[T, ...]` instead of `typing.List`, `typing.Dict`, `typing.Set`, `typing.Tuple`.
* **Self-Referencing Types**: Use `from typing import Self` (Python 3.11+) instead of `TypeVar('T', bound=...)` for method chaining or factory methods.
* **Annotated Types**: Use `Annotated[T, ...]` for attaching validation, metadata, or dependency injection hints.
* **Type Aliases**: Use the `type` statement in Python 3.12+ (or `TypeAlias` in earlier versions):
  ```python
  # Python 3.12+
  type ImageTensor = torch.Tensor
  type BoundingBox = tuple[float, float, float, float]
  ```

### Example
```python
from pathlib import Path
from typing import Self

class ConfigLoader:
    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.settings: dict[str, int | float | str] = {}

    def load(self) -> Self:
        # Load configuration
        return self
```

---

## 2. Data Modeling: Dataclasses vs Pydantic

* Use `@dataclass(slots=True, kw_only=True)` for internal lightweight data transfer objects (DTOs) and high-performance in-memory containers.
* Use **Pydantic v2** (`pydantic.BaseModel`) when parsing external inputs, configs, API payloads, or validating complex invariants.

### Dataclass Best Practice
```python
from dataclasses import dataclass, field

@dataclass(slots=True, kw_only=True)
class DetectionResult:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    metadata: dict[str, str] = field(default_factory=dict)
```

### Pydantic v2 Best Practice
```python
from pydantic import BaseModel, Field, field_validator

class ModelConfig(BaseModel):
    model_name: str = Field(min_length=1)
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=1e-3, gt=0.0)

    @field_validator("model_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip().lower()
```

---

## 3. Structural Pattern Matching (`match` / `case`)

Use Python 3.10+ structural pattern matching for routing actions, message dispatching, and destructuring complex data:

```python
match command:
    case {"action": "detect", "image_path": str(path)}:
        run_detection(path)
    case {"action": "train", "epochs": int(epochs)} if epochs > 0:
        start_training(epochs)
    case _:
        raise ValueError(f"Unknown command: {command}")
```

---

## 4. Modern Tooling & Packaging

### Tooling Stack
* **Package & Virtualenv Management**: Use `uv` for ultra-fast dependency resolution and virtualenv creation.
* **Project Metadata**: Maintain standard `pyproject.toml` (PEP 517 / PEP 621).
* **Linter & Formatter**: Use `ruff` (`ruff check . --fix`, `ruff format .`) to replace Black, Flake8, and isort.
* **Static Type Checker**: Use `mypy` with `--strict` mode or standard type checks.

### Standard `pyproject.toml` Structure
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-cv-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "opencv-python-headless>=4.9.0",
    "numpy>=1.26.0",
    "pydantic>=2.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

---

## 5. Clean Idioms & Anti-Patterns

1. **Pathlib over `os.path`**:
   - Do: `from pathlib import Path; path = Path("data") / "images"`
   - Avoid: `os.path.join("data", "images")`
2. **Context Managers for Resources**:
   - Always manage files, sockets, locks, and GPU resource contexts with `with` blocks.
3. **Explicit Exceptions**:
   - Never use bare `except:`. Always catch specific exceptions (`except (ValueError, KeyError) as exc:`).
4. **Avoid Mutable Default Arguments**:
   - Do: `def fn(items: list[str] | None = None) -> list[str]: ...`
   - Avoid: `def fn(items: list[str] = []) -> list[str]: ...`
