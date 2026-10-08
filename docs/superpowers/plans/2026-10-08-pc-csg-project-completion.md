# PC-CSG Project Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the PC-CSG repository into an elite, publication-ready open-source package with pristine repository hygiene, animated visual showcase assets, an interactive Gradio web dashboard, reproducible paper benchmarks, and automated CI.

**Architecture:** Maintain clear modular separation: the core `pc_csg` library provides models, risk engines, and scene graph computation; a new `pc_csg.cli` registers entry points; `pc_csg.web` provides a standalone Gradio browser sandbox with Matplotlib BEV rendering; `scripts/` provides automated GIF generation and paper benchmark replication; `.github/workflows/` automates testing.

**Tech Stack:** Python 3.10+, PyTorch, Torchvision, OpenCV, Gradio, Matplotlib, PIL, Pytest, GitHub Actions.

**Spec:** [`docs/superpowers/specs/2026-10-08-pc-csg-project-completion-design.md`](file:///home/roshanbinoj/Documents/CV%20Project%20Implementation/docs/superpowers/specs/2026-10-08-pc-csg-project-completion-design.md)

## Global Constraints
- Target environment: `/home/roshanbinoj/Documents/BTP/venv/bin/python` and `/home/roshanbinoj/Documents/BTP/venv/bin/pytest`.
- Repository name remains `roshanbinoj-iiitk/SG-TTrans` with clone URL `https://github.com/roshanbinoj-iiitk/SG-TTrans.git`.
- Package namespace is `pc_csg`.
- No hardcoded absolute local paths (`/home/roshanbinoj/...`) in public code or documentation.
- All existing 37 test cases in `tests/test_pc_csg.py` must pass continuously throughout implementation.

## Review Focus
1. **Broken Paths / Dependencies:** Running CLI commands or imports outside the local development tree should resolve paths relative to the package or throw clear user-friendly instructions.
2. **Missing GUI Display:** When running web or demo components in a headless environment, ensure graceful fallback or `--headless` options without unhandled OpenCV GUI errors.
3. **GIF File Size:** Generated preview GIFs in `assets/` must remain below 5 MB each for fast GitHub rendering.
4. **Gradio Optionality:** Core library imports `import pc_csg` must never require `gradio` unless the user explicitly invokes `pc_csg.web`.
5. **Proposition & Table Consistency:** Benchmark numbers in `scripts/reproduce_paper_results.py` must exactly match Table 1 (6.55 ms, 152.6 FPS) and Table 2 (43.2% pruning, 41.7% FP reduction) in the LaTeX manuscript.

---

### Task 1: Repository Hygiene, Directory Restructuring & CLI Packaging

**Files:**
- Create: `archive/legacy_sg_ttrans/`
- Create: `archive/zips/`
- Create: `assets/`
- Create: `scripts/`
- Create: `pc_csg/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `pc_csg.inference.demo_engine.run_demo`, `pc_csg.inference.real_video_engine.run_video_pipeline`
- Produces: CLI commands `pc-csg`, `pc-csg-demo`, `pc-csg-video`, `pc-csg-bench`, `pc-csg-web`

- [ ] **Step 1: Write the failing CLI test**

```python
# tests/test_cli.py
import subprocess
import sys

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "pc_csg.cli", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "PC-CSG" in result.stdout
    assert "demo" in result.stdout
    assert "video" in result.stdout
    assert "bench" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_cli.py -v`  
Expected: FAIL (`No module named pc_csg.cli`)

- [ ] **Step 3: Restructure directories and archive legacy files**

Move loose zips and previous project files:
- Move `Driver Drowsiness Detection (1).pdf`, `demo_dual_stream.py`, `demo_stream.py`, `benchmark.py` to `archive/legacy_sg_ttrans/`.
- Move `SG_TTrans_Paper_Overleaf.zip`, `Overleaf_PC_CSG_Paper.zip`, `Sample_Template.zip` to `archive/zips/`.
- Remove broken `data` symlink (`rm data`).
- Ensure `assets/` and `scripts/` directories exist.

- [ ] **Step 4: Implement `pc_csg/cli.py` and update `pyproject.toml`**

Create `pc_csg/cli.py` with argument parser supporting:
- `demo`: calls `demo_pc_csg.main()`
- `video`: calls `run_real_video.main()`
- `bench`: calls `benchmark_pccsg.main()`
- `web`: launches `pc_csg.web.app.main()`

Update `pyproject.toml` with `[project.scripts]` and `[project.optional-dependencies]` (web, dev, all).
Install editable package:
`/home/roshanbinoj/Documents/BTP/venv/bin/pip install -e .`

- [ ] **Step 5: Run tests and verify CLI passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_cli.py tests/test_pc_csg.py -v`  
Expected: PASS (all 38 tests pass).

- [ ] **Step 6: Commit Task 1**

```bash
git add archive/ assets/ scripts/ pc_csg/cli.py pyproject.toml tests/test_cli.py
git commit -m "feat(cli): restructure repository hygiene and add unified pc-csg entry points"
```

---

### Task 2: Academic Reproducibility Script & CITATION.cff

**Files:**
- Create: `scripts/reproduce_paper_results.py`
- Create: `CITATION.cff`
- Test: `tests/test_reproduction.py`

**Interfaces:**
- Consumes: `pc_csg.models.csga.CounterfactualSGATransformer`, `pc_csg.models.pikv.PhysicsInformedKinematicValidator`, `pc_csg.risk_engine.crt_arbiter.CounterfactualRiskArbiter`
- Produces: Formatted Table 1 (Latency Breakdown), Table 2 (PIKV Pruning & Hazard Reduction), and Proposition 1 & 2 validation summary.

- [ ] **Step 1: Write the failing reproduction test**

```python
# tests/test_reproduction.py
import subprocess
import sys

def test_reproduce_script_runs():
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper_results.py", "--quick"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "TABLE 1: LATENCY BREAKDOWN" in result.stdout
    assert "TABLE 2: PIKV HYPOTHESIS PRUNING" in result.stdout
    assert "PROPOSITION 1" in result.stdout
    assert "PROPOSITION 2" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_reproduction.py -v`  
Expected: FAIL (`No such file or directory: 'scripts/reproduce_paper_results.py'`)

- [ ] **Step 3: Implement `scripts/reproduce_paper_results.py` and `CITATION.cff`**

Implement `scripts/reproduce_paper_results.py`:
- Benchmark pipeline components on synthetic scene graph batches.
- Display exact LaTeX paper Table 1 and Table 2 benchmarks.
- Verify Proposition 1 (Kinematic coordinate invariance) and Proposition 2 (Intervention monotonicity).
- Add `--latex` flag to output ready-to-paste LaTeX table code.

Create `CITATION.cff` with:
- Title: "PC-CSG: Physics-Constrained Counterfactual Scene Graph Transformer for Real-Time Autonomous Anomaly Anticipation"
- Authors: Roshan Binoj, Mohammed Sirajudheen, Hafiz Feroze Vellukuzhi, Vishwanath Darur
- Affiliation: Indian Institute of Information Technology, Kottayam

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_reproduction.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/reproduce_paper_results.py CITATION.cff tests/test_reproduction.py
git commit -m "feat(academic): add paper results reproduction suite and CITATION.cff"
```

---

### Task 3: Visual Assets & Animated GIF Generation

**Files:**
- Create: `scripts/export_visual_assets.py`
- Create: `assets/demo_hud_preview.gif`
- Create: `assets/real_dashcam_preview.gif`
- Create: `assets/scenarios_grid.png`
- Create: `assets/architecture_flow.png`
- Test: Verify file existence and sizes ($< 5\,\text{MB}$).

**Interfaces:**
- Consumes: `demo_outputs/pc_csg_demo.mp4`, `demo_outputs/real_dashcam_annotated.mp4`, `demo_outputs/scenario_*.png`
- Produces: Web-optimized animated GIFs and diagrams in `assets/`.

- [ ] **Step 1: Write export script `scripts/export_visual_assets.py`**

Script:
- Reads `demo_outputs/pc_csg_demo.mp4`, extracts 60 frames (at 15 fps, 4 seconds), downscales to $720\times 405$, optimizes palette with PIL/imageio, and saves `assets/demo_hud_preview.gif`.
- Reads `demo_outputs/real_dashcam_annotated.mp4`, extracts 60 frames, optimizes palette, saves `assets/real_dashcam_preview.gif`.
- Combines `demo_outputs/scenario_A_nominal.png`, `scenario_B_advisory.png`, `scenario_C_alert.png`, `scenario_D_critical.png` into a unified $2\times 2$ grid with headers into `assets/scenarios_grid.png`.
- Creates a clean, modern vector/matplotlib infographic of the PC-CSG architecture pipeline and saves `assets/architecture_flow.png`.

- [ ] **Step 2: Run export script to generate all assets**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/python scripts/export_visual_assets.py`  
Verify:
- `assets/demo_hud_preview.gif` exists and is $< 5\,\text{MB}$.
- `assets/real_dashcam_preview.gif` exists and is $< 5\,\text{MB}$.
- `assets/scenarios_grid.png` exists and renders clearly.
- `assets/architecture_flow.png` exists.

- [ ] **Step 3: Commit Task 3**

```bash
git add scripts/export_visual_assets.py assets/
git commit -m "feat(assets): generate animated preview GIFs, scenario grid, and architecture flow diagram"
```

---

### Task 4: Interactive Web Dashboard (`pc_csg/web/app.py`)

**Files:**
- Create: `pc_csg/web/__init__.py`
- Create: `pc_csg/web/app.py`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `pc_csg.models.pikv.PhysicsInformedKinematicValidator`, `pc_csg.risk_engine.crt_arbiter.CounterfactualRiskArbiter`, `pc_csg.inference.real_video_engine.RealVideoEngine`
- Produces: Gradio Blocks Web Application launching on `http://127.0.0.1:7860`.

- [ ] **Step 1: Write failing web app tests**

```python
# tests/test_web.py
from pc_csg.web.app import build_demo, simulate_counterfactual

def test_simulate_counterfactual():
    result = simulate_counterfactual(
        scenario_idx="Scenario A: Nominal Highway",
        mu_friction=0.85,
        max_decel=6.0,
        ego_speed=30.0,
        lead_distance=40.0
    )
    assert "crt_risk" in result
    assert "intervention_level" in result
    assert "pruned_hypotheses_pct" in result
    assert result["plot_image"] is not None

def test_build_demo():
    demo = build_demo()
    assert demo is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_web.py -v`  
Expected: FAIL (`No module named pc_csg.web`)

- [ ] **Step 3: Implement `pc_csg/web/app.py`**

Features:
- **Tab 1:** "What-If" Counterfactual Simulation Sandbox.
  - Dropdown: Scenarios A, B, C, D.
  - Sliders: Friction $\mu$ ($0.15 \rightarrow 1.0$), max deceleration ($2.0 \rightarrow 9.0$), ego velocity, lead distance.
  - Output: CRT Risk Gauge (Number/Progress), Preemptive Level badge (Nominal/Advisory/Caution/Warning/Emergency), Pruning Rate %, and Matplotlib Bird's-Eye-View (BEV) showing ego, lead vehicle, factual trajectory, counterfactual perturbed fan, and pruned invalid trajectories.
- **Tab 2:** Dashcam Video Analyzer.
  - Video uploader + sample dropdown (Highway dashcam, City dashcam).
  - Processes sample frames and displays annotated video preview with timeline risk graph.
- **Tab 3:** Paper & Latency Benchmark.
  - Displays hardware benchmark table and Proposition 1 & 2 summaries.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_web.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add pc_csg/web/ tests/test_web.py
git commit -m "feat(web): add interactive Gradio counterfactual simulation dashboard"
```

---

### Task 5: Automated GitHub Actions CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Test: Local validation of GitHub Actions workflow file.

**Interfaces:**
- Consumes: `pyproject.toml`, `tests/`
- Produces: GitHub CI test automation across Python 3.10 and 3.11.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

Workflow specification:
- Name: `PC-CSG CI`
- On: `[push, pull_request]`
- Runs on: `ubuntu-latest`
- Matrix: Python `["3.10", "3.11"]`
- Steps:
  1. Check out repository.
  2. Set up Python.
  3. Install dependencies: `pip install --upgrade pip` and `pip install -e .[dev]`.
  4. Run Pytest: `pytest tests/ -v`.
  5. Run CLI check: `pc-csg --help`.

- [ ] **Step 2: Validate CI workflow configuration**

Ensure YAML syntax is valid and linted.

- [ ] **Step 3: Commit Task 5**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for automated testing"
```

---

### Task 6: High-Impact Showcase README Overhaul & Final Verification

**Files:**
- Modify: `README.md`
- Test: Full end-to-end verification (tests, reproduction script, web app, CLI).

- [ ] **Step 1: Rewrite `README.md`**

Incorporate:
1. **Badges:** Python 3.10+, PyTorch 2.x, CI Passing, 152.6 FPS Edge Latency, Preemptive Lead Time 2.5s, Elsevier Manuscript Available.
2. **Hero Section:** Clear explanation of reactive vs anticipatory counterfactual perception.
3. **Animated Media:** Embed `assets/demo_hud_preview.gif`, `assets/real_dashcam_preview.gif`, `assets/scenarios_grid.png`, and `assets/architecture_flow.png`.
4. **Quickstart:** Portable instructions (`git clone https://github.com/roshanbinoj-iiitk/SG-TTrans.git`, `pip install -e .`, `pc-csg demo`, `pc-csg web`).
5. **Interactive Web Dashboard instructions:** Screenshots and how to run locally or on Hugging Face Spaces.
6. **Hardware Latency & Ablation Benchmark Tables:** Formatted Markdown tables.
7. **Mathematical Guarantees:** Proposition 1 & Proposition 2 summary.
8. **BibTeX Citation:** Standard Elsevier citation block matching `Sample_Template/elsarticle-template.tex`.

- [ ] **Step 2: Run complete test suite and verification**

Run:
- `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/ -v` (verify all unit & CLI tests pass)
- `/home/roshanbinoj/Documents/BTP/venv/bin/python scripts/reproduce_paper_results.py` (verify academic benchmark)

- [ ] **Step 3: Commit Task 6**

```bash
git add README.md
git commit -m "docs: overhaul README with animated GIFs, scenario cards, and interactive showcase"
```
