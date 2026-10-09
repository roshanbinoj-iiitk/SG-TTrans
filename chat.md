

Log in
You said:

what all skills should I set up in antigravity Skip to content
Home
eBooks
Blog
About
Apps

Go back
The 10 Claude Code Skills I Actually Use at Work
Posted on:
April 8, 2026
Welcome, Developer 👋

I want to be upfront about something: I’ve written a few posts on Claude Code skills already. Curated lists, ranked by stars, organized by category. Useful as a starting point, maybe. But none of them answer the question I actually get asked the most, which is: what do you personally have installed?

So here it is. These are the 10 skills I’m running right now, across the projects I work on day to day — a React Native platform serving a large student user base, and a side project built on MySQL and Metabase. Real work. Real context. No filler.

1. superpowers
Source: github.com/obra/superpowers
Stars: ~40,900


/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
This one changed how I work more than anything else on this list. Before Superpowers, I’d drop a task into Claude Code and it would start writing immediately. Sometimes that worked. More often, it went in a direction I didn’t want and I’d spend time unwinding it.

Superpowers forces a different sequence: clarify → spec → plan → execute → review. The test-driven-development skill inside it is strict — it won’t write implementation before a failing test exists. The plan-eng-review step locks in architecture and data flow before any code is touched.

I use this for every feature that takes more than a session to build. It’s overkill for small fixes. For anything real, it’s the skill I’d keep if I had to keep only one.

2. vercel-react-best-practices
Source: Vercel official — VoltAgent/awesome-agent-skills
Installs: 176,400 via skills.sh


npx skills@latest add vercel/react-best-practices
I spend most of my day in React Native and Expo. This skill isn’t Native-specific, but the React fundamentals it encodes — component structure, data fetching patterns, error boundaries — transfer directly.

The concrete difference: before this skill, Claude would generate components that worked in isolation and broke in context. After installing it, the output follows patterns I’d actually write myself. Proper hook usage, clean prop interfaces, no mystery side effects.

If React is part of your stack in any form, this belongs at user level.

3. frontend-design
Source: Anthropic official — github.com/anthropics/skills — already bundled
Installs: 124,100

This one’s already in your install and most people never touch it. I didn’t for the first two months.

The difference it makes is less dramatic than the others but it’s consistent: Claude stops defaulting to the same visual output every time. Inter font, purple gradient, grid cards — that’s the statistical center of every UI Claude has seen. This skill pulls it away from that center and toward something that looks considered rather than generated.

I use it mostly for internal tools, reports, and dashboards where the design was never specified and someone needs to ship something that doesn’t look terrible.

4. spartan-ai-toolkit
Source: github.com/spartan-stratos/spartan-ai-toolkit


npx @c0x12c/ai-toolkit@latest --local
Quality gates. That’s the reason I keep this installed. It runs typecheck → lint → test → review in sequence and won’t move to the next step if the previous one fails. That sounds obvious. In practice, it changes everything.

Without quality gates, Claude Code will write code, notice a failing test, patch the test to pass, and present that as done. With this toolkit, that path doesn’t exist. The gates force the right order.

I also use the React stack profile specifically, which gives Claude React and TypeScript conventions rather than generic JavaScript advice. Eight profiles are available — Go, Python, Java, Kotlin, React, and more. Pick yours.

5. mattpocock/skills
Source: github.com/mattpocock/skills
Skills: 17 dev workflow skills — install individually:


npx skills@latest add mattpocock/skills/write-a-prd
npx skills@latest add mattpocock/skills/request-refactor-plan
npx skills@latest add mattpocock/skills/git-guardrails-claude-code
# full list at the repo
Matt Pocock runs Total TypeScript. If you write TypeScript, you’ve probably read something he’s written. His skills reflect the same TypeScript-first perspective: strict type discipline, clean interface boundaries, and refactoring patterns that don’t just make the code shorter but make it more honest about what it does.

The ones I use most: tdd (stricter than Claude’s default, closer to what I’d actually enforce in a PR review) and refactoring-plan (generates a step-by-step plan for a refactor before touching anything). The refactoring one in particular has saved me from making things worse in the name of making them better.

6. planetscale
Source: PlanetScale official — planetscale.com/docs


npx skills@latest add planetscale/skills
Database work is where I’ve seen Claude make its worst mistakes. Not syntax errors — those are obvious. The expensive kind: schema decisions that seem fine and cause pain six months later, queries that run cleanly at 100 rows and fall apart at 100,000, missing indexes that only show up in production.

Without this skill, Claude treats your database like any other code. It writes something that runs and moves on. The PlanetScale skill changes that default. It gives Claude the branching model (one database branch per feature, merge when done, never touch production schema directly), plus the query performance fundamentals that matter: index design, N+1 detection, query plan awareness.

I work with MySQL day to day and the skill earns its place even if you’re not on PlanetScale specifically. The schema discipline and query hygiene it encodes transfer directly to any MySQL-compatible database. It’s the most complete SQL skill available right now.

7. web-design-guidelines
Source: github.com/vercel-labs/web-interface-guidelines
Stars: 22,000 · Weekly installs: 133,400


npx skills@latest add vercel-labs/web-interface-guidelines
Where frontend-design improves what Claude generates, this skill reviews what you’ve already built. It fetches Vercel’s interface guidelines and checks your code against them — outputs a list of violations with specifics.

I use it at the end of a UI session, not the beginning. Think of it as a linter that catches the things ESLint doesn’t touch: visual hierarchy issues, inconsistent spacing, accessibility patterns that are technically fine but practically terrible.

The 133K weekly installs tell you it’s not just me.

8. doc-coauthoring
Source: Anthropic official — github.com/anthropics/skills


/plugin marketplace add anthropics/skills
/plugin install example-skills@anthropic-agent-skills
Every engineering team has documentation debt. At UC, I work with infrastructure docs, runbooks, and the occasional disaster recovery plan. On the side project side, there are specs, data dictionaries, and API docs that are always half-written.

The doc-coauthoring skill changes how Claude approaches any writing task: it starts by transferring context, building an outline, getting sign-off on structure, then drafting section by section. The output reads like a human wrote it because the process required a human to guide it.

I’ve used this for a full DR plan, multiple technical specs, and a handful of onboarding guides. Every time, the result was something I was comfortable putting my name on.

9. code-simplifier
Source: Anthropic official — github.com/anthropics/claude-plugins-official


/plugin marketplace add anthropics/claude-plugins-official
/plugin install code-simplifier@claude-plugins-official
AI moves fast and writes faster. The code it produces works but accumulates complexity quickly — nested ternaries, functions doing three things at once, abstractions that made sense in the moment and don’t six weeks later.

The code-simplifier skill is focused on one thing: take recently modified code and make it cleaner without changing what it does. It follows your project’s conventions from CLAUDE.md, applies them consistently, and flags clarity issues like nested conditionals and overly compact logic in favour of explicit, readable alternatives.

The rule it enforces that I appreciate most: never change behaviour, only how behaviour is expressed. It won’t refactor your logic or suggest architecture changes — it cleans up the code you just wrote and moves on. That constraint is what makes it trustworthy.

10. TerraShark
Source: github.com/LukasNiessen/terrashark


/plugin marketplace add LukasNiessen/terrashark
/plugin install terrashark
I manage Azure infrastructure as part of my day job and Terraform is where I’ve seen Claude make the worst mistakes — not obvious ones, but the kind that are fine at first and painful six months later. Missing lifecycle rules, hardcoded values that should be variables, modules that don’t compose cleanly.

Most Terraform skills dump a wall of reference text into context on every activation — 4,400+ tokens — and then leave Claude to figure out what applies. TerraShark is different. The core SKILL.md is 79 lines and costs around 600 tokens on activation. Instead of front-loading docs, it runs a diagnostic sequence: capture context → identify failure mode → load only the relevant references → propose fixes with explicit risk controls → validate → deliver structured output.

It’s grounded in HashiCorp’s official recommended practices. When anything conflicts, it sides with HashiCorp. That’s the right call.

What I’d install first if I was starting over
If I had to pick four and nothing else for the first week:

superpowers — the workflow foundation, everything else builds on top of this
spartan-ai-toolkit — quality gates between every step, won’t let Claude skip ahead
planetscale — if you touch a database at all, install this before you write another query
TerraShark — same logic, but for infrastructure
code-simplifier comes in week two — once you have a pattern for writing code, you want something that keeps it clean without second-guessing the logic. frontend-design and web-design-guidelines follow naturally if UI is part of your work. mattpocock/skills if TypeScript is your daily language.

Conclusion
The point isn’t to install everything. It’s to start with what matches the work you actually do and let the gaps tell you what to add next.

Stay focused, Developer!

  Claude  AI
Copyright © 2026
 | 
All rights reserved.
for a python computer vision project

ChatGPT said:
For a Python computer-vision project, I would not copy that article's 10 skills wholesale. Most of them are React/TypeScript/SQL/Terraform-oriented.

I’d build your Antigravity setup around Python quality + CV/ML + experiment discipline.

My recommended stack
1. superpowers — workflow foundation
Keep this from the article.

Useful when a CV task is substantial: dataset changes, model architecture changes, training pipelines, refactors, etc. The clarify → plan → implement → test/review workflow is particularly useful because CV projects can accumulate experimental code very quickly.

Install:
obra/superpowers

2. modern-python — Python foundation
I'd make this your Python baseline.

It covers modern Python project setup around uv, Ruff, and type checking, rather than letting the agent invent a different project structure every time. 
O
officialskills.sh

Install:

npx skills add https://github.com/trailofbits/skills --skill modern-python

This is more relevant to you than the React skills in the article.

3. computer-vision-opencv — CV-specific knowledge
This is the most directly relevant specialist skill I found.

It covers OpenCV, PyTorch, image processing, object detection, feature detection, video processing, etc. 
E
explainx.ai
+1

Install:

npx skills add https://github.com/mindrally/skills --skill computer-vision-opencv

For a project involving cv2, image/video pipelines, augmentation, detection or segmentation, I'd definitely have a CV-specific skill.

4. python-testing — testing
CV projects need this more than people expect.

You want tests for things like:

image preprocessing

coordinate/bounding-box transformations

dataset loading

augmentation behavior

tensor shapes

model output contracts

post-processing

video frame handling

edge cases

The current Python testing skill ecosystem has guidance around pytest, fixtures, parameterization, unit/integration tests and property-based testing. 
L
LazySkills

One option:

npx skills add mindrally/skills@python-testing

5. python-performance-optimization — very useful for CV
This is one I'd add specifically because you're doing computer vision.

CV code can be correct but painfully slow because of:

unnecessary image copies

Python loops over pixels

CPU↔GPU transfers

repeated preprocessing

inefficient dataloaders

excessive image decoding

bad batch sizes

unnecessary tensor conversions

The current Python skill catalog specifically includes a performance workflow focused on profiling CPU/memory and optimizing the measured hot path rather than guessing. 
L
LazySkills

6. code-simplifier — after you've established the code
I'd keep this from the article, but not as a core skill.

It's useful after an agent has generated a training/data/inference pipeline and you want to reduce accidental complexity without changing behavior.

What I would NOT install from that article
For a Python CV project, these are generally irrelevant unless your project happens to use them:

Skill	For your CV project
vercel-react-best-practices	❌ Skip
frontend-design	❌ Skip unless building a UI
web-design-guidelines	❌ Skip unless building a web frontend
planetscale	❌ Skip unless using PlanetScale/MySQL
TerraShark	❌ Skip unless using Terraform
mattpocock/skills	🟡 Mostly skip; TypeScript-oriented
doc-coauthoring	🟡 Optional
spartan-ai-toolkit	🟡 Potentially useful, but I'd first establish a Python-specific quality gate

The setup I'd actually use
If this were my Python CV repo, my initial stack would be:

superpowers
modern-python
computer-vision-opencv
python-testing
python-performance-optimization
code-simplifier

And then add skills based on the actual project:

PyTorch / deep learning
        ↓
computer-vision-opencv
        ↓
python-testing
        ↓
python-performance-optimization
        ↓
modern-python
        ↓
superpowers

If you're doing serious ML training
I'd also consider a PyTorch-specific skill rather than relying entirely on a generic CV skill. NVIDIA's current TAO ecosystem, for example, has agent-oriented workflows around training, fine-tuning and optimizing computer-vision models, although that makes more sense if you're actually using the NVIDIA/TAO ecosystem. 
N
NVIDIA Docs

And if your project is specifically YOLO/object detection, segmentation, OCR, face recognition, multi-object tracking, etc., I'd tailor the skills further rather than installing a giant generic collection.

One important thing
Don't optimize for having 20 skills installed.

For an agent, I'd rather have 5–7 highly relevant skills than 30 overlapping ones. The current skills ecosystem itself has separate workflows for Python structure, quality, testing, dependencies, performance, error handling, etc., which is a good reason to compose a focused stack instead of installing every "Python" skill you find. 
L
LazySkills
+1

If you tell me what your CV project actually does (e.g. YOLO object detection, medical imaging, OCR, face recognition, video tracking, image classification, OpenCV-only, PyTorch training, etc.), I can give you the exact Antigravity skills to install, with the commands, plus which ones should be global vs project-local.




O
E
L
Sources
Log in for personalized, step-by-step guidance.


Log in

Sign up for free
You said:

Core Focus: Real-Time Driver Drowsiness Detection & ADAS Risk Monitoring
The project implements an in-cabin computer vision framework titled:

SG-TTrans (Spatial-Geometric Temporal Transformer) for Real-Time Driver Drowsiness Detection and Risk-Aware Autonomous Safety Monitoring.

Instead of a single-frame detector or standard CNN, it is a multi-modal, spatio-temporal video monitoring system designed for edge deployment in vehicles.

How the Pipeline Works
Dual-Stream Visual & Kinematic Feature Extraction:

Spatial Appearance Stream: Uses a lightweight CNN backbone (MobileNetV4) on $224 \times 224$ RGB facial frames to extract dense 256-D visual tokens that remain robust against cabin lighting changes, glares, and shadows.
Geometric Kinematics Stream: Uses MediaPipe FaceMesh to track 3D landmarks and extract 16-D kinematic metrics:
EAR (Eye Aspect Ratio) & PERCLOS (Percentage of Eyelid Closure)
MAR (Mouth Aspect Ratio) for yawning detection
3D Head Pose Euler angles (pitch, yaw, roll) for head-nodding and distraction detection (robust against sunglasses and masks).
Cross-Modal Fusion & Positional Encoding:

Operates over a sliding temporal window of 60 frames @ 30 FPS (2.0s duration).
Linearly projects and fuses appearance tokens with geometric kinematics and 1D learnable temporal positional embeddings.
Temporal Dynamic Decay Attention (TDDA) Transformer:

A 4-block Temporal Transformer with a custom attention kernel (TDDA).
Filters out voluntary transient blinks ($< 250\text{ ms}$) via exponential distance decay.
Amplifies sustained micro-sleep episodes ($> 400\text{ ms}$) and yawning sequences.
Dual Output Heads:

Driver State Classification: Multi-class classification (Alert, Drowsy, Micro-Sleep, Yawn, Distracted) trained with multi-class Focal Loss.
Risk-Aware Safety Index ($RSI$): A continuous risk engine $RSI(t) \in [0.0, 1.0]$ synthesizing visual fatigue persistence with vehicle telematics (vehicle speed $v(t)$ and radar forward Time-to-Collision / $TTC$).
Graduated ADAS Interventions: Triggers stepped safety actions (Level 0: Safe $\rightarrow$ Level 1: Visual warning $\rightarrow$ Level 2: Audio/haptic alerts $\rightarrow$ Level 3: Active lane assist / automated braking).

ChatGPT said:
Given that architecture, I’d tailor the Antigravity setup much more heavily toward PyTorch, computer vision, temporal modeling, testing, profiling, and edge deployment.

Your project is not a generic Python project. It has several failure-prone areas: video preprocessing, landmark geometry, temporal datasets, transformer attention, multimodal fusion, GPU/CPU boundaries, real-time latency, and safety-critical-ish intervention logic.

The skills I would set up
Tier 1 — Install these first
superpowers

Planning complex features

TDD workflow

Breaking research ideas into implementable tasks

Reviewing implementation before moving on

Definitely install.

modern-python

uv

Ruff

typing

project structure

dependency management

Python conventions

Definitely install.

Computer Vision / OpenCV skill

OpenCV

image/video processing

frame extraction

geometric operations

camera pipelines

preprocessing

Definitely install.

PyTorch skill

This is the big one missing from the article for your project.

Your agent needs to understand:

nn.Module

custom attention

Transformer blocks

tensor shapes

Dataset / DataLoader

mixed precision

CUDA

model checkpoints

training/evaluation loops

inference

GPU memory

torch.compile

profiling

For SG-TTrans, I'd consider this mandatory.

Python testing / pytest

Especially important for your geometry and temporal logic.

I'd want tests around:

EAR calculation
MAR calculation
head-pose calculation
PERCLOS calculation
temporal window construction
TDDA attention
RSI calculation
state transitions
intervention thresholds
tensor dimensions
model output contracts

Python performance optimization

This is unusually important for your project because your target is real-time edge inference.

You ultimately care about:

camera
   ↓
face detection
   ↓
FaceMesh
   ↓
MobileNetV4
   ↓
fusion
   ↓
TDDA
   ↓
classification + RSI
   ↓
intervention

The agent should be encouraged to measure latency rather than blindly optimize.

Tier 2 — I'd add these specifically for your architecture
7. ML experiment / research workflow
I'd want a skill that forces the agent to distinguish:

experiment
≠
production code

For example:

experiments/
    exp_001_baseline/
    exp_002_tdda/
    exp_003_multimodal/
    exp_004_ablation/

with reproducible:

seeds

configs

checkpoints

metrics

datasets

hyperparameters

experiment names

This matters enormously when you're comparing:

CNN baseline
vs
CNN + temporal model
vs
CNN + geometry
vs
CNN + geometry + TDDA

You don't want the agent accidentally turning an experiment into undocumented production behavior.

8. Data / dataset engineering
This should be a dedicated skill if you're building your own dataset.

Your agent should understand:

train/validation/test splitting

subject-independent splitting

temporal leakage

class imbalance

frame sampling

sequence construction

augmentation

corrupted video handling

missing frames

label synchronization

Subject leakage is particularly important for driver monitoring.

You don't want:

Driver A
video 1 → training
video 2 → validation

because the model can partially learn the driver's identity/appearance rather than generalizable drowsiness features.

9. Metrics / ML evaluation
I'd strongly recommend a dedicated evaluation workflow.

Your classification metrics shouldn't stop at accuracy.

You should be able to automatically produce:

macro F1
weighted F1
per-class precision
per-class recall
confusion matrix
ROC/PR where applicable
false alarm rate
miss rate
latency
FPS
GPU memory

And for the temporal component:

detection delay
micro-sleep detection delay
false alarm duration
event-level precision/recall

For your project, event-level evaluation is arguably more meaningful than simply counting individual frames.

Tier 3 — Very important for your particular project
10. Edge / inference optimization
Because your stated goal is:

Real-Time ... Edge Deployment

I'd make this a first-class skill.

The agent should know about:

ONNX export

TensorRT

FP16

INT8

dynamic/static shapes

model quantization

inference benchmarking

CPU vs GPU execution

memory copies

batching vs streaming

pipeline parallelism

camera capture latency

You eventually want a benchmark like:

Component                  Latency
────────────────────────────────────
FaceMesh                   8.2 ms
MobileNetV4                4.7 ms
Feature fusion             0.2 ms
TDDA                       1.8 ms
Classification             0.1 ms
RSI                        0.02 ms
────────────────────────────────────
Total                      14.9 ms
Effective FPS              67.1

Rather than simply:

"The model runs in real time."

11. code-simplifier
Keep this from the article.

But I'd run it after model implementation rather than letting it interfere with experimental research code.

Particularly useful for:

forward()

and preprocessing/inference pipelines that tend to become enormous.

12. Documentation / technical writing
doc-coauthoring is actually useful for your project.

You have enough architectural novelty that you will eventually need:

README
architecture.md
dataset.md
training.md
evaluation.md
deployment.md
model-card.md
risk-engine.md

I'd use the documentation skill for those.

What I would NOT install
For this project, I'd skip:

❌ vercel-react-best-practices
❌ frontend-design
❌ web-design-guidelines
❌ mattpocock/skills
❌ planetscale
❌ TerraShark

Unless your surrounding application happens to use those technologies.

Your ideal Antigravity skill stack
I'd organize it roughly like this:

                    SG-TTrans
                       │
             ┌─────────┴─────────┐
             │                   │
         Development          ML Research
             │                   │
      modern-python        experiment workflow
             │             dataset engineering
        superpowers        evaluation/metrics
             │
      ┌──────┴───────┐
      │              │
   PyTorch       Computer Vision
      │              │
      └──────┬───────┘
             │
       Temporal ML
             │
       TDDA Transformer
             │
      Multimodal Fusion
             │
       ┌─────┴─────┐
       │           │
   Testing      Profiling
       │           │
       └─────┬─────┘
             │
       Edge Deployment
       ONNX / TensorRT
             │
             ▼
       Real-time system

My priority order
If you don't want to install everything immediately:

Phase 1 — right now
1. superpowers
2. modern-python
3. PyTorch
4. computer-vision / OpenCV
5. pytest / Python testing

Phase 2 — once the model works
6. ML experiment management
7. dataset engineering
8. ML evaluation / metrics
9. code-simplifier

Phase 3 — before claiming "real-time"
10. performance optimization
11. edge inference / ONNX / TensorRT
12. deployment / profiling

One
ChatGPT said:
Unable to connect
•

Retry

No file chosenNo file chosenNo file chosen

Chat with ChatGPT
Ask ChatGPT



