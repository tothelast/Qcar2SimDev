# Report Improvement Plan

## Context

The thesis report (`report/`) is structurally complete across 9 chapters and 4 appendices, but has critical gaps:
- **Chapter 7 (Results)** is entirely placeholder stubs with all values marked "TBD"
- **Chapter 2 (Background)** describes SimLingo at a high level but lacks neural network architecture depth
- **Chapter 5 (Inference)** mentions command mode but doesn't explain why it was skipped
- No figures or graphs are included anywhere in the report
- The ACC baseline (`results/test_acc_baseline.py`) doesn't produce metrics comparable to the SimLingo test framework

The goal is to: (1) deepen the SimLingo architecture explanation, (2) clarify target-point vs command mode scope, (3) run full experiments on both controllers, (4) populate Chapter 7 with real data and graphs.

---

## Task 1: Expand SimLingo Architecture in Chapter 2

**File to modify:** `report/chapters/02_background.tex`

**Current state:** Section 2.3 (SimLingo) is 3 sentences. Section 2.4 (InternVL2-1B) is 2 sentences. No architecture diagrams, no discussion of encoders/embeddings/adaptors.

**What to add** (expand section 2.3 into multiple subsections):

### 1.1 Vision Encoder
- InternViT-300M-448px from InternVL2-1B
- Dynamic high-resolution processing: input images split into 448x448 tiles, encoded independently
- Pixel unshuffle downsampling (factor 4): each tile → 256 visual tokens
- With N_I=2 tiles: 512 visual tokens total, output shape e_I ∈ R^((N_I·256)×D)
- **Source:** `simlingo/pretrained/InternVL2-1B/modeling_intern_vit.py`, `simlingo/simlingo_training/utils/internvl2_utils.py`

### 1.2 Navigational Conditioning Embeddings
- Target-point mode: GPS waypoints → MLP → 2 navigational embeddings e_nav ∈ R^(2×D)
- Command mode: text tokenizer → variable-length embeddings e_nav ∈ R^(N_HLC×D)
- **Source:** `simlingo/simlingo_training/models/adaptors/adaptors.py` (WaypointInputAdaptor: Linear(2→256)→ReLU→Linear(256→512)→ReLU→Linear(512→D))

### 1.3 Token Interleaver and Prompt Construction
- Global prompt template: `"<image features>\n Current speed: <v>m/s. Command: <nav features>. <task prompt>."`
- Token interleaver (IL) replaces placeholder tokens with actual embeddings
- Four task prompt variants: driving, commentary, VQA, action dreaming
- Unified input sequence: visual tokens + nav embeddings + text tokens + learnable action query tokens

### 1.4 Dual-Head Action Prediction
- Speed waypoints (temporal): future coordinates at 0.25s intervals → PID longitudinal control
- Path/route waypoints (geometric): coordinates at 1m intervals → PID lateral control
- Both predicted non-autoregressively via learnable query tokens q_p, q_w
- MLP heads on LLM output features: Linear(hidden→512)→SiLU→Linear(512→256)→SiLU→Linear(256→2)
- Cumulative sum converts deltas to absolute positions
- **Source:** `simlingo/simlingo_training/models/adaptors/adaptors.py` (DrivingAdaptor class)

### 1.5 Language Generation Head
- Autoregressive next-token prediction from LLM hidden states
- Cross-entropy loss for language tokens
- Optional: commentary, QA, action dreaming tasks

### 1.6 Training Losses
- Smooth-L1 loss on speed waypoints and route waypoints
- Cross-entropy loss on language tokens
- Multi-task loss with configurable weights

**Reference:** SimLingo paper (Renz et al., CVPR 2025) — arxiv.org/abs/2503.09594

---

## Task 2: Clarify Target-Point Mode Scope

**Files to modify:** `report/chapters/05_inference.tex`, `report/chapters/04_training.tex`

**Current state:** Chapter 5 §5.3 mentions both modes but command mode is marked in red text. Chapter 4 notes "no natural-language commentary supervision" but doesn't explicitly state why command mode was excluded.

**What to add:**

### In Chapter 4 (Training/Data Collection):
Add a paragraph in §4.1 explaining:
- Original SimLingo trains with both target-point and command mode, randomly switching during training
- Command mode requires language annotations mapping routes to HLC strings ("turn left", "follow the road", etc.)
- QLabs data collection pipeline (`data_collection/data_recorder.py`) uses constant `command=4` ("follow the road") for all samples
- No natural-language route annotation was collected because: (a) QLabs lacks a route planner with semantic HLC labels, (b) collecting language data was out of scope for this work
- Consequence: fine-tuned model only supports target-point conditioning

### In Chapter 5 (Inference):
- Remove the red text marking on command mode
- Rewrite §5.3 to clearly state: "This work uses target-point mode exclusively" with a forward reference to §4.1 for the rationale
- Note that the inference code (`inference/simlingo_model.py`) supports both modes but command mode was not evaluated

---

## Task 3: Adapt ACC Baseline for Comparable Metrics

**File to create:** `results/test_acc_baseline_roundabout.py`
**Reference:** `results/test_acc_baseline.py` (existing simple ACC), `results/test_simlingo_roundabout.py` (target metrics framework)

The current ACC baseline (`test_acc_baseline.py`) is a simple single-run script that:
- Uses pure pursuit steering + LiDAR obstacle detection
- Stops when obstacle detected, no metrics computed
- No structured JSON output, no pass/fail criteria

**What to build:** A new test script that runs the ACC baseline through the **same test matrix** as the SimLingo framework:

### Test Matrix (same as SimLingo):
- Baseline (no obstacle): 5 runs
- Obstacle variants 1–5: 2 runs each
- Total: 15 runs

### Metrics to compute (reuse functions from `test_simlingo_roundabout.py`):
- `compute_safety_metrics()` — collision detection, stopping distance
- `compute_route_coverage()` — % of waypoints reached within 1.5m
- `compute_lateral_deviation()` — avg and max deviation from centerline
- `compute_distance_traveled()` — total path length
- `detect_stuck()` — stuck detection
- `determine_pass_status()` — same pass/fail criteria

### Implementation approach:
1. Import metric functions from `test_simlingo_roundabout.py` (or extract to shared module `results/metrics.py`)
2. Replace the SimLingo model inference with the ACC controller logic from `test_acc_baseline.py`:
   - Pure pursuit steering via `compute_steering()`
   - LiDAR obstacle detection via `get_obstacle_distance()`
   - Simple speed control: cruise at 2.0 m/s, stop when obstacle detected
3. Use same `TestScenario`, `TestResult`, `SafetyMetrics` dataclasses
4. Output same JSON/CSV format as SimLingo framework
5. Trajectory logging with same fields (step, timestamp, position, heading, speed, etc.)

---

## Task 4: Run Full Experiment Suite

**Prerequisite:** QLabs must be running with SDCS RoadMap loaded.

### 4.1 Run Full SimLingo Test Suite (15 runs)
```bash
python results/test_simlingo_roundabout.py \
  --checkpoint simlingo/outputs/2025_11_26_18_06_21_qlabs_roundabout_finetune/checkpoints/epoch_14.pt
```
- Produces: `results/test_results_<timestamp>.json` and `.csv`
- Trajectory logs in `results/runs/`

### 4.2 Run Full ACC Baseline Test Suite (15 runs)
```bash
python results/test_acc_baseline_roundabout.py
```
- Produces: `results/acc_test_results_<timestamp>.json` and `.csv`
- Trajectory logs in `results/acc_runs/`

---

## Task 5: Generate Graphs

**File to create:** `results/generate_report_figures.py`

### 5.1 Existing Graphs (copy to report/figures/)
- `policy_vs_expert_curve.png` — ADE across training epochs (from `plot_results.py`)
- `metrics.png` — Training loss curves (6-panel: total loss, route loss, speed WPs loss, language loss, LR, epoch progress)

### 5.2 New Graphs to Generate

#### Graph 1: Route Coverage Comparison (Bar Chart)
- X-axis: scenario names (baseline, obstacle_var1–5)
- Y-axis: route coverage %
- Two bars per scenario: SimLingo (blue) vs ACC Baseline (orange)
- Error bars from multiple runs
- **Data source:** Both test_results JSON files

#### Graph 2: Safety Metrics Comparison (Grouped Bar Chart)
- Metrics: collision rate, stop success rate, avg stopping distance
- Grouped by controller (SimLingo vs ACC)
- Only obstacle scenarios

#### Graph 3: Lateral Deviation Comparison (Bar Chart)
- X-axis: scenarios
- Y-axis: avg lateral deviation (meters)
- SimLingo vs ACC baseline bars

#### Graph 4: Trajectory Overlay Plot
- Bird's-eye view of route with:
  - Route centerline (blue dashed)
  - SimLingo trajectories (solid lines, different colors per run)
  - ACC baseline trajectories (dotted lines)
  - Obstacle positions (red markers)
- One plot per obstacle variant (or a representative subset)
- **Data source:** trajectory_log.json files

#### Graph 5: Pass/Fail Summary Table (as figure)
- Scenario × Controller matrix showing pass/fail with color coding
- Include aggregate pass rates

### 5.3 Graph Generation Script Structure
```python
# results/generate_report_figures.py
# Loads both JSON result files
# Generates all figures and saves to report/figures/
# Usage: python results/generate_report_figures.py \
#   --simlingo-results results/test_results_<ts>.json \
#   --acc-results results/acc_test_results_<ts>.json \
#   --output-dir report/figures/
```

---

## Task 6: Populate Chapter 7 (Results)

**File to modify:** `report/chapters/07_results_stub.tex` → rename to `report/chapters/07_results.tex`
**Update:** `report/main.tex` to reference new filename

### Structure:

#### §7.1 Training Convergence
- Reference `metrics.png` (training loss curves) — shows convergence in ~5 epochs
- Reference `policy_vs_expert_curve.png` (ADE curve) — ADE drops from 0.447 to 0.120 over 15 epochs (73% improvement)
- Discuss: rapid initial improvement, diminishing returns after epoch 10, final ADE of 0.12 means predictions average 12cm from expert trajectories

#### §7.2 Baseline Route Following (No Obstacle)
- Table: SimLingo vs ACC baseline on clean roundabout
- Metrics: route coverage, avg/max lateral deviation, completion time
- Both should achieve ~100% route coverage; compare tracking precision
- **Data from:** 5 baseline runs per controller

#### §7.3 Obstacle Avoidance Results
- Main results table (Table 7.1):

| Scenario | Controller | Coverage (%) | Collision | Stopped | Stop Dist (m) | Avg Lat Dev (m) |
|----------|-----------|-------------|-----------|---------|---------------|-----------------|
| obstacle_var1 | SimLingo | X | Y | Z | ... | ... |
| obstacle_var1 | ACC | X | Y | Z | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

- Pass/fail analysis per scenario
- Discussion of where each controller succeeds/fails

#### §7.4 Comparative Analysis
- Bar chart figures comparing both controllers
- Trajectory overlay figures for representative scenarios
- Key findings: SimLingo uses vision to detect obstacles vs ACC uses LiDAR; trade-offs in reaction distance, path smoothness

#### §7.5 Discussion
- Strengths: SimLingo can navigate without LiDAR, learned behavior generalizes to obstacle positions
- Weaknesses: lower route coverage in obstacle scenarios (stops early vs ACC's precise detection range)
- Limitations of current evaluation: in-distribution testing only, single route topology

---

## Task 7: Update Supporting Sections

### 7.1 Abstract (`report/frontmatter/abstract.tex`)
- Remove red text placeholder about quantitative results
- Add 1-2 sentences summarizing actual findings (pass rates, ADE)

### 7.2 Chapter 8 Limitations (`report/chapters/08_limitations.tex`)
- Update "Incomplete quantitative results" bullet to reference actual results
- Keep other limitations as-is

### 7.3 Chapter 9 Conclusion (`report/chapters/09_conclusion.tex`)
- Replace placeholder references to incomplete results with actual summary

### 7.4 Appendix D (`report/appendices/open_questions.tex`)
- Mark "Final quantitative results" as RESOLVED
- Note date of experiment execution

---

## Implementation Order

1. **Extract shared metrics module** → `results/metrics.py` (from test_simlingo_roundabout.py)
2. **Create ACC baseline test framework** → `results/test_acc_baseline_roundabout.py`
3. **Run SimLingo full test suite** (15 runs, ~40 min with QLabs)
4. **Run ACC baseline full test suite** (15 runs, ~20 min with QLabs)
5. **Create graph generation script** → `results/generate_report_figures.py`
6. **Generate all figures** → `report/figures/`
7. **Expand Chapter 2** (SimLingo architecture deep-dive)
8. **Update Chapters 4 & 5** (target-point mode clarification)
9. **Write Chapter 7** (replace stubs with real results)
10. **Update abstract, conclusion, limitations, appendix D**
11. **Copy plan to** `report/plan.md`
12. **Build PDF** to verify: `cd report && latexmk -pdf -interaction=nonstopmode main.tex`

---

## Verification

1. **Scripts run without error:**
   - `python results/test_acc_baseline_roundabout.py` completes 15 runs
   - `python results/generate_report_figures.py` produces all figures
2. **Figures exist in `report/figures/`:**
   - `policy_vs_expert_curve.png`, `metrics.png`
   - `route_coverage_comparison.png`, `safety_comparison.png`
   - `lateral_deviation_comparison.png`, `trajectory_overlay_*.png`
   - `pass_fail_summary.png`
3. **LaTeX compiles:** `cd report && latexmk -pdf main.tex` produces clean PDF
4. **Chapter 7 has no TBD values** — all metrics populated from actual JSON data
5. **No red text** remaining in the document

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `report/chapters/02_background.tex` | Expand SimLingo architecture |
| `report/chapters/04_training.tex` | Add command mode rationale |
| `report/chapters/05_inference.tex` | Clarify target-point only scope |
| `report/chapters/07_results_stub.tex` | Replace with full results |
| `results/test_simlingo_roundabout.py` | SimLingo test framework (existing) |
| `results/test_acc_baseline.py` | ACC baseline logic (existing, reuse) |
| `results/test_acc_baseline_roundabout.py` | New: ACC comparable test framework |
| `results/metrics.py` | New: shared metrics module |
| `results/generate_report_figures.py` | New: graph generation |
| `plot_results.py` | Existing ADE plot script |
| `tools/plot_wandb_metrics.py` | Existing training metrics plot |
| `outputs/.../ade_results_all_epochs.json` | ADE data (epoch 0: 0.447 → epoch 14: 0.120) |
| `simlingo/simlingo_training/models/` | Architecture reference for Ch. 2 |
