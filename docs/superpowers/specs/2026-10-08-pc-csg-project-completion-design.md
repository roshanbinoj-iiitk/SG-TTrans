# Design Specification: PC-CSG Project Completion & Showcase Architecture

**Date:** 2026-10-08  
**Topic:** PC-CSG Repository Polish, Showcase Media, Interactive Web Dashboard, and Academic Reproducibility  
**Repository:** `roshanbinoj-iiitk/SG-TTrans`  
**Package Name:** `pc_csg`  

---

## 1. Executive Summary & Objective

The objective of this design is to elevate the **PC-CSG** (*Physics-Constrained Counterfactual Scene Graph Transformer*) project from an active research repository into a world-class, top-tier open-source project and academic deliverable.

The target audiences are **Elsevier journal reviewers**, **BTP defense evaluators/professors**, and **open-source AI/CV researchers and engineers**.

### Success Criteria:
1. **Repository Hygiene & Portability:** Zero broken symlinks, zero hardcoded machine paths (`/home/roshanbinoj/...`), legacy files cleanly archived into `archive/`, and standardized CLI entry points via `pip install -e .`.
2. **Instant Visual Impact:** Crisp animated GIFs (`assets/demo_hud_preview.gif`, `assets/real_dashcam_preview.gif`), dynamic badges, architecture diagrams, and scenario matrices embedded directly in `README.md`.
3. **Interactive Web Dashboard:** A standalone Gradio browser application (`pc_csg/web/app.py` / `pc-csg web`) enabling one-click scenario testing, real-time friction $\mu$ manipulation, CRT risk gauge meters, and dashcam video analysis without requiring a local display server.
4. **Academic Synchronization & Reproducibility:** A single-command reproducer (`scripts/reproduce_paper_results.py`) validating mathematical propositions (Propositions 1 & 2) and reproducing Table 1 (latency breakdown) and Table 2 (hypothesis pruning), accompanied by a formal `CITATION.cff`.
5. **Continuous Integration:** A GitHub Actions workflow (`.github/workflows/ci.yml`) executing all 37 pytest test suites across Python versions to display a green passing status badge.

---

## 2. Structural Architecture & Directory Layout

### 2.1 Before vs. After Directory Organization

```
Current (Cluttered):
├── Driver Drowsiness Detection (1).pdf    # Legacy
├── SG_TTrans_Paper_Overleaf.zip           # Legacy
├── Overleaf_PC_CSG_Paper.zip              # Legacy zip
├── Sample_Template.zip                    # Legacy zip
├── data -> /home/... (broken symlink)     # Broken link
├── demo_dual_stream.py                    # Legacy script
├── demo_stream.py                         # Legacy script
├── benchmark.py                           # Legacy script
├── demo_pc_csg.py
├── run_real_video.py
├── train_real_world.py
├── pc_csg/
└── README.md (hardcoded local paths)

Target Architecture (Polished & Professional):
├── .github/
│   └── workflows/
│       └── ci.yml                         # Automated pytest CI
├── archive/                               # Cleanly preserved legacy assets
│   ├── legacy_sg_ttrans/
│   │   ├── Driver Drowsiness Detection (1).pdf
│   │   ├── demo_dual_stream.py
│   │   ├── demo_stream.py
│   │   └── benchmark.py
│   └── zips/
│       ├── SG_TTrans_Paper_Overleaf.zip
│       ├── Overleaf_PC_CSG_Paper.zip
│       └── Sample_Template.zip
├── assets/                                # Media & visual assets for GitHub
│   ├── demo_hud_preview.gif               # Looping HUD demo
│   ├── real_dashcam_preview.gif           # Real road dashcam demo
│   ├── scenarios_grid.png                 # 4-scenario comparison grid
│   └── architecture_flow.png              # High-res pipeline diagram
├── pc_csg/                                # Core Python library
│   ├── models/                            # CSGA & PIKV models
│   ├── risk_engine/                       # CRT & arbiter
│   ├── scene_graph/                       # Spatio-temporal scene graph
│   ├── inference/                         # Real-time inference & video engine
│   ├── training/                          # Loss functions & trainer
│   ├── web/                               # Interactive Gradio web dashboard
│   │   ├── __init__.py
│   │   └── app.py
│   ├── cli.py                             # Unified CLI commands
│   ├── config.py
│   └── types.py
├── scripts/                               # Reproducibility & media export tools
│   ├── reproduce_paper_results.py         # One-click Table 1 & 2 generator
│   └── export_visual_assets.py            # Automated GIF & screenshot generator
├── tests/                                 # PyTorch & Kinematic test suite
│   └── test_pc_csg.py                     # 37 verified test cases
├── Sample_Template/                       # Elsevier manuscript LaTeX source
├── CITATION.cff                           # Academic citation metadata
├── pyproject.toml                         # Modern packaging with CLI scripts
└── README.md                              # Showcase README with badges & GIFs
```

---

## 3. Pillar-by-Pillar Technical Specifications

### 3.1 Pillar 1: Repository Hygiene & Packaging (`pyproject.toml`)

1. **Move legacy artifacts:** Relocate all `.zip` archives, the legacy drowsiness detection PDF, and old dual-stream scripts to `archive/`.
2. **Remove broken symlinks:** Remove `data` symlink. Ensure all scripts default to local sample files or throw helpful descriptive errors if external data is missing.
3. **Packaging & CLI Commands:**
   Update `pyproject.toml` with console scripts:
   ```toml
   [project.scripts]
   pc-csg = "pc_csg.cli:main"
   pc-csg-demo = "pc_csg.cli:main_demo"
   pc-csg-video = "pc_csg.cli:main_video"
   pc-csg-web = "pc_csg.web.app:main"
   pc-csg-bench = "pc_csg.cli:main_bench"
   ```
   This allows anyone who runs `pip install -e .` to immediately execute `pc-csg demo` without worrying about Python paths.
4. **Optional Dependency Groups:**
   ```toml
   [project.optional-dependencies]
   dev = ["pytest>=7.0.0", "ruff>=0.1.0"]
   web = ["gradio>=4.0.0", "matplotlib>=3.7.0"]
   all = ["pytest>=7.0.0", "gradio>=4.0.0", "matplotlib>=3.7.0", "ultralytics>=8.0.0"]
   ```

---

### 3.2 Pillar 2: Interactive Web Dashboard (`pc_csg/web/app.py`)

A clean, modern **Gradio** web application providing an intuitive UI without requiring a local GUI/X11 display:

#### Features:
1. **Interactive "What-If" Counterfactual Sandbox:**
   * **Preset Dropdown:** Nominal Cruising, Lead Braking, Truck Swerve, Emergency Collision.
   * **Physics Sliders:**
     * Road friction $\mu \in [0.10, 1.00]$ (dry asphalt, wet rain, snow, ice).
     * Maximum deceleration $a_{\max} \in [2.0, 9.0]\,\text{m/s}^2$.
     * Ego velocity $v_{\text{ego}} \in [10, 45]\,\text{m/s}$ ($36 - 162\,\text{km/h}$).
   * **Live Interactive Visuals:**
     * **CRT Risk Gauge:** Continuous meter showing calculated risk $0.0 \rightarrow 1.0$.
     * **5-Level Preemptive Intervention Indicator:** Nominal (Green) $\rightarrow$ Advisory (Cyan) $\rightarrow$ Caution (Yellow) $\rightarrow$ Warning (Orange) $\rightarrow$ Emergency AEB (Red).
     * **Kinematic Constraint Pruning Summary:** Percentage of hypotheses pruned by Coulomb friction circles.
     * **Bird’s-Eye-View (BEV) Trajectory Plot:** Matplotlib plot rendering factual trajectory vs counterfactual perturbations and safety boundaries.

2. **Dashcam Video Risk Analyzer:**
   * File uploader for MP4/AVI videos, with quick-select buttons for pre-loaded sample road videos.
   * Runs YOLOv8 + PC-CSG inference on video frames.
   * Renders the processed video with HUD overlay alongside a synchronized timeline risk graph.

3. **Academic Inspector & Benchmark:**
   * Live latency benchmarks, theoretical proposition checks, and links to the paper.

---

### 3.3 Pillar 3: Visual Assets & README Showcase

1. **GIF Generation Tool (`scripts/export_visual_assets.py`):**
   * Uses `moviepy` or `imageio` / `cv2` to extract high-quality, lightweight GIFs:
     * `assets/demo_hud_preview.gif` (~2-4 MB, smooth 15 fps loop).
     * `assets/real_dashcam_preview.gif` (~2-4 MB, showing real road detection).
   * Generates a 4-panel scenario comparison grid `assets/scenarios_grid.png` displaying Nominal, Advisory, Alert, and Critical interventions.
2. **README Overhaul:**
   * **Badges:** Python 3.10+, PyTorch 2.x, 37 Tests Passing, 152.6 FPS Edge Latency, Preemptive Lead Time 2.5s, Elsevier Manuscript.
   * **Hero Section:** Clear value proposition: "Why Anticipatory Counterfactual Perception Beats Reactive Perception".
   * **Visual Cards & GIFs:** Embedded immediately below the fold.
   * **Latency Benchmark Table:** Formatted comparison of pipeline stages against baselines.
   * **Quickstart:** Clean, copy-pasteable commands without machine-specific paths.
   * **Citation Block:** Elsevier bibtex entry.

---

### 3.4 Pillar 4: Academic Reproducibility & Paper Sync

1. **Reproducibility Script (`scripts/reproduce_paper_results.py`):**
   * Automatically executes the pipeline over test batches.
   * Formats and prints:
     * **Table 1: Computational Latency Breakdown** (Scene Graph 0.24ms, CSGA 1.35ms, Bicycle Model 4.33ms, PIKV 0.50ms, CRT Arbiter 0.14ms $\rightarrow$ Total: 6.55ms / 152.6 FPS).
     * **Table 2: Kinematic Pruning Efficiency** (43.2% invalid counterfactual hypotheses pruned by PIKV, 41.7% false-positive intervention reduction).
     * **Proposition Checks:** Validates Proposition 1 (Kinematic Invariance under coordinate transform) and Proposition 2 (Convergence of Preemptive Intervention).
2. **Citation (`CITATION.cff`):**
   * Standardized citation metadata for GitHub's native "Cite this repository" button.

---

### 3.5 Pillar 5: Continuous Integration (CI/CD)

1. **GitHub Actions (`.github/workflows/ci.yml`):**
   * Triggers on `push` and `pull_request` to `main`.
   * Matrix tests on Ubuntu with Python 3.10 and 3.11.
   * Installs headless dependencies (`opencv-python-headless`, `torch`, `torchvision`, `pytest`).
   * Runs `pytest tests/test_pc_csg.py -v`.
   * Guarantees green CI badge in the README.

---

## 4. Verification and Validation Strategy

| Feature / Artifact | Verification Method | Success Metric |
| :--- | :--- | :--- |
| **Directory Hygiene** | Run `git status`, test imports | Zero broken symlinks; all legacy files in `archive/` |
| **Packaging & CLI** | `pip install -e .` followed by `pc-csg --help` | All CLI subcommands execute cleanly |
| **Unit Tests** | `pytest tests/test_pc_csg.py` | All 37 tests pass |
| **Web Dashboard** | Run `pc-csg web` or `python -m pc_csg.web.app` | Browser opens, sliders update CRT and BEV plot in $<50\,\text{ms}$ |
| **Visual Assets** | Inspect generated `.gif` and `.png` in `assets/` | File size $<5\,\text{MB}$, crisp resolution, smooth loop |
| **README Rendering** | Markdown preview | Badges render, GIFs animate, clone URLs valid |
| **Paper Reproduction** | `python scripts/reproduce_paper_results.py` | Prints Tables 1 & 2 matching Elsevier manuscript |
| **CI Workflow** | GitHub Actions syntax check via `action-lint` or dry-run | Valid YAML, passes test run |

---

## 5. Risk Assessment & Mitigation

1. **Large GIF File Sizes:** Large GIFs slow down GitHub page loading.  
   *Mitigation:* Optimize frame rate (15 FPS), scale resolution to $720\times 405$, and keep duration to 3-4 seconds per loop.
2. **Missing GUI Display in Headless Environments:** Web app or demo commands failing if X11/wayland display is absent.  
   *Mitigation:* Use `opencv-python-headless` in CI and web mode, ensure `--headless` flag remains supported for CLI commands.
3. **Gradio Dependency Bloat:** Gradio shouldn't be mandatory for users only running the core PyTorch model.  
   *Mitigation:* Place `gradio` inside `[project.optional-dependencies] web` so core library remains ultra-lightweight.
