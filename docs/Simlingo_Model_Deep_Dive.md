# Simlingo Driving Model – Complete Walkthrough

This guide explains the full Simlingo training and inference pipeline as implemented in this repository. It is intentionally detailed and visual so that someone new to the project (or to multimodal driving models in general) can follow every step.

---

## 1. From Raw Files to a Training Sample

Each training example is built from three on-disk assets collected in the simulator:

```
TownXYZ/
├── rgb/
│   └── 0000.jpg          ← camera frame (1024×512, BGR on disk)
├── measurements/
│   └── 0000.json.gz      ← vehicle & route metadata
└── results.json.gz       ← route quality filter (score & infractions)
```

### 1.1 `results.json.gz`

`dataset_base.py` first checks this file to make sure the run is valid (perfect score or allowed infractions). Invalid routes are skipped before any expensive processing happens.

### 1.2 `measurements/{seq}.json.gz`

This gzip-compressed JSON contains everything the model needs:

| Field | Meaning | Where it is used |
|-------|---------|------------------|
| `ego_matrix` | 4×4 pose matrix for this frame | Used to convert future positions into the ego frame (`dataset_base.py:get_waypoints`) |
| `route_original`, `route` | Future navigation path (world & adjusted) | Resampled to 20 route points (`dataset_base.py:load_route`) |
| `target_point`, `target_point_next` | Short-horizon navigation waypoints | Inserted into the language prompt & placeholder tensors (`dataset_base.py:get_navigational_conditioning`) |
| `speed` | Instantaneous vehicle speed | Shown in the prompt; stored as tensor in `DrivingInput` |
| `command`, `next_command` | High-level driving command (optional) | Used when `route_as` includes `"command"` |
| `augmentation_*` | Camera augmentation metadata | Applied if image shift augmentation is enabled |

### 1.3 `rgb/{seq}.jpg`

The raw camera frame. `load_images` (in `dataset_base.py`) reads it with OpenCV, optionally augments/crops it, and converts it to an RGB tensor.

---

## 2. Building the Python Objects

`Data_Driving.__getitem__` (in `dataset_driving.py`) combines those files into a structured dictionary. During batching, `DataModule.dl_collate_fn` (in `datamodule.py`) turns that dictionary into a `DrivingExample`, which is the core container passed to the model:

```
DrivingExample
├── driving_input (DrivingInput)
│   ├── camera_images      ← processed image patches [B, T, N, C, H, W]
│   ├── vehicle_speed      ← speed tensor [B, 1]
│   ├── target_point       ← [[x1, y1], [x2, y2]] placeholder values
│   ├── prompt             ← tokenized conversation (LanguageLabel)
│   └── prompt_inference   ← alt prompt used during inference decoding
└── driving_label (DrivingLabel)
    ├── waypoints          ← 10 future ego-frame waypoints (2D)
    ├── path               ← 20 resampled route points (2D)
    ├── answer             ← ground-truth language string (LanguageLabel)
    └── image_ff_org       ← original-size image tensor (unused by loss)
```

> **Why 10 waypoints?**  
> The dataloader loads `hist_len + pred_len` measurements, converts them to ego-frame coordinates, and keeps `waypoints[1:-1]`. With defaults (`hist_len=1`, `pred_len=11`), that leaves 10 future targets spaced 0.2 seconds apart.

---

## 3. Forward Pass Overview

```
┌────────────────────────────────────────────────────────────────┐
│            DrivingModel.forward_loss (training mode)            │
├────────────────────────────────────────────────────────────────┤
│ 1. AdaptorList.forward:                                         │
│    - LanguageAdaptor.forward → prompt embeddings                │
│    - DrivingAdaptor.forward  → 10 speed queries (+20 route)     │
│    - Vision encoder injects image patches as placeholder tokens │
│                                                                  │
│ 2. LanguageModel.forward (LoRA-enabled Qwen variant):            │
│    - consumes concatenated embeddings (image + prompt + queries)│
│    - returns hidden states + logits                             │
│                                                                  │
│ 3. Adaptor losses:                                               │
│    - DrivingAdaptor: produces waypoint & route predictions       │
│      and compares with driving_label.waypoints/path              │
│    - LanguageAdaptor: computes cross-entropy on answer tokens    │
│                                                                  │
│ 4. Loss aggregation:                                             │
│    - Smooth L1 (waypoints)                                       │
│    - Smooth L1 (route, optional)                                 │
│    - Cross-entropy (language)                                    │
│    - Sum → backprop (updates only LoRA weights)                  │
└────────────────────────────────────────────────────────────────┘
```

### 3.1 Vision Encoder (Frozen)

`VLMEncoderModel` wraps InternVL. Because `freeze=True` in training configs, all parameters are set to `requires_grad = False` except the small MLP used for placeholder replacement. This means fine-tuning focuses on the language model’s LoRA adapters.

### 3.2 Language Adaptor

- Tokenizes the prompt (`LanguageLabel.phrase_ids`).
- Applies placeholder substitution so tokens like `<TARGET_POINT>` can be replaced with numeric embeddings representing the actual target values (`vision_model.image_encoder.replace_placeholder_tokens`).
- Produces input embeddings (shape `[B, tokens, hidden]`) and corresponding masks.

### 3.3 Driving Adaptor

- Adds learned query embeddings:
  - `future_speed_waypoints = 10` → predicts near-term motion.
  - `future_waypoints = 20` (only when `predict_route_as_wps=True`) → predicts longer path.
- Each head is a small MLP ending with cumulative sums, turning predicted offsets into absolute coordinates (`adapters.py:169-178`).

---

## 4. Losses in Detail

```
                Predictions                          Labels
   ┌────────────────────────────┐        ┌──────────────────────────┐
   │ speed_wps: [10, 2]         │  vs    │ driving_label.waypoints  │
   │ route:     [20, 2] (opt.)  │  vs    │ driving_label.path       │
   │ language logits            │  vs    │ driving_label.answer     │
   └────────────────────────────┘        └──────────────────────────┘

Loss = SmoothL1(speed_wps, waypoints)
     + SmoothL1(route, path)        (if enabled)
     + CrossEntropy(language_logits, answer_tokens)
```

- **Smooth L1** is used for stability; it behaves like L2 near zero and like L1 for large errors.
- **Cross-entropy** uses a loss mask so prompt tokens do not contribute, only the answer text.
- The final scalar loss is summed by `summarise_losses` and reported to Lightning, which drives optimizer updates.

> **Note:** When `speed_wps_mode='1d'`, the driving adaptor expects distances instead of 2D coordinates. In the default `'2d'` configuration used here, it keeps the full X/Y offsets.

---

## 5. Inference Flow

Inference reuses the same pieces but switches to evaluation mode:

```
┌───────────────────────────────────────────────────────┐
│ DrivingModel.forward(driving_input)                   │
├───────────────────────────────────────────────────────┤
│ 1. Build adaptor embeddings (prompt + image + queries)│
│ 2. (Optional) greedy language generation              │
│ 3. Concatenate generated tokens with driving queries  │
│ 4. Run language model forward → get hidden states     │
│ 5. DrivingAdaptor.get_predictions → waypoints         │
│ 6. Return:                                            │
│      speed_wps [10×2], route [20×2], language string  │
└───────────────────────────────────────────────────────┘
```

- `SimlingoModelWrapper` takes care of assembling `DrivingInput` from live camera frames (`inference/simlingo_model.py`).
- The wrapper also trims the language string to remove the trailing `"Waypoints:"` placeholder text so downstream code gets a clean narration.

---

## 6. Worked Example

Assume `measurements/0000.json.gz` looks like this (simplified):

```json
{
  "ego_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "route": [[0.0, 0.0], [1.0, 0.4], ..., [19.0, 9.5]],
  "target_point": [12.3, 2.1],
  "target_point_next": [13.1, 2.6],
  "speed": 5.5,
  "command": 4,
  "next_command": 4
}
```

1. **Prompt construction** (`dataset_driving.py`):
   - `"Current speed: 5.5 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints."`
2. **Answer template**:
   - `"Waypoints:"`
3. **Waypoints tensor**:
   - `[ [1.0, 0.0], [1.8, 0.2], ..., [10.2, 1.8] ]` (10 entries)
4. **Route tensor**:
   - `[ [0.0, 0.0], [1.0, 0.4], ..., [19.0, 9.5] ]` (resampled to 20 points)

During training the model might predict:

```
pred_speed_wps = [[0.9, 0.1], [1.7, 0.3], ..., [9.8, 1.7]]
pred_route     = [[0.1, 0.0], [1.1, 0.5], ..., [18.8, 9.4]]
pred_tokens    = "Waypoints:"
```

Losses compare these directly with the stored tensors and text. If everything matches perfectly, the loss is zero; otherwise the gradients adjust the LoRA layers.

---

## 7. Key Takeaways

1. **All data originates from the CARLA/QLabs recording format** and is loaded lazily at training time.
2. **`DrivingInput` and `DrivingLabel` completely define each sample**, ensuring the model always knows:
   - what it saw (images, prompt, target point, speed)
   - what it should predict (future trajectory + narration)
3. **Only LoRA adapters are updated**, keeping the heavy vision/language backbones frozen for efficient fine-tuning.
4. **Inference reuses the same heads**—the outputs you see at runtime are exactly the supervised targets used during training.

Armed with this map of the code, you can trace any value from disk to loss function and back. Don’t hesitate to open the referenced files side-by-side while reading; the comments and section headers in `dataset_base.py`, `dataset_driving.py`, and `models/adaptors/adaptors.py` line up with the descriptions in this document.
