# Simlingo Training Flow: Simple Explanation

## Overview

The model learns by comparing its predictions to ground truth data. It receives **images + text prompts** as input and produces **waypoints + text descriptions** as output. Only the language model's LoRA weights get updated.

---

## The Complete Flow (One Training Step)

### **Step 1: Load Data** 
📁 `dataloader/dataset_driving.py` → `__getitem__()`

**What happens:** Load one sample from disk
```
Input files:
├── rgb/0000.jpg                    → Camera image
├── measurements/0000.json          → Speed, target point, waypoints
└── results.json.gz                 → Route metadata

Output (DrivingExample):
├── camera_images: [1, 3, 1024, 512]     # Image tensor
├── vehicle_speed: 5.2                    # m/s
├── target_point: [10.5, 2.3]            # Navigation target
├── prompt: "You are a driving assistant..." # Text input
└── ground_truth:
    ├── waypoints: [[1,0], [2,0], ...]   # Where to drive
    └── answer: "Driving straight. Waypoints: [[1,0], [2,0], ...]"  # Text description
```

---

### **Step 2: Convert Prompt to Embeddings**
📁 `models/adaptors/adaptors.py` → `LanguageAdaptor.forward()` (line 256)

**What happens:** Convert text to numbers the model understands
```python
# Text prompt (example)
prompt = "You are a driving assistant. Target waypoint: <TARGET_POINT><TARGET_POINT>. Describe the scene."

# Tokenize to IDs
ids = [1523, 403, 264, 7193, ...]  # Each word → number

# Convert to embeddings
embeddings = embed_tokens(ids)  # Shape: [1, 150, 2048]
                                # 150 tokens, each is 2048-dim vector
```

---

### **Step 3: Process Images Through Vision Encoder**
📁 `models/driving.py` → `forward_model()` (line 200-205)

**What happens:** Extract visual features (vision encoder is **FROZEN**)
```python
# Input image: [1, 3, 1024, 512]
visual_features = vision_encoder(camera_images)  # No gradients!

# Output: [1, 256, 2048]  
# 256 image patches, each is 2048-dim vector
```

**Key:** Vision encoder weights don't change. It just extracts features.

---

### **Step 4: Combine Visual Features + Text Embeddings**
📁 `models/driving.py` → `forward_model()` (line 208-214)

**What happens:** Concatenate image features and text embeddings
```python
# Visual features:    [1, 256, 2048]  (from frozen vision encoder)
# Text embeddings:    [1, 150, 2048]  (from prompt)
# Combined:           [1, 406, 2048]  (256 + 150 tokens)

input_embeds = concat(visual_features, text_embeddings)
```

---

### **Step 5: Language Model Processes Everything**
📁 `models/driving.py` → `forward_model()` (line 215-233)

**What happens:** Language model reads visual+text features and generates output
```python
# Input: [1, 406, 2048] combined features
features, logits = language_model(input_embeds)

# Output:
# - features: [1, 406, 2048]  # Hidden representations
# - logits:   [1, 406, 151936] # Predictions for next token (vocab size = 151936)
```

**Key:** Language model has LoRA weights that **ARE being trained**.

---

### **Step 6: Extract Predictions**
📁 `models/adaptors/adaptors.py` → `DrivingAdaptor.get_predictions()`

**What happens:** Decode features into waypoints and text
```python
# From language model output features:
predicted_waypoints = driving_head(features)  # [[1.1, 0.05], [2.2, 0.1], ...]
predicted_tokens = argmax(logits)             # [1523, 403, 264, ...]
predicted_text = decode(predicted_tokens)     # "Driving straight. Waypoints: ..."
```

---

### **Step 7: Compute Losses**
📁 `models/adaptors/adaptors.py` → `compute_loss()`

**What happens:** Compare predictions to ground truth

#### **7a. Waypoint Loss** (line 209)
```python
# Predicted: [[1.1, 0.05], [2.2, 0.1], [3.1, 0.15], ...]
# Ground truth: [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], ...]

waypoint_loss = smooth_l1_loss(predicted_waypoints, ground_truth_waypoints)
# Result: 0.15 (lower is better)
```

#### **7b. Language Loss** (line 271-273)
```python
# Predicted tokens: [1523, 403, 264, 7193, ...]
# Ground truth:     [1523, 403, 264, 7194, ...]  (slightly different)

language_loss = cross_entropy(predicted_tokens, ground_truth_tokens)
# Result: 2.34 (lower is better)
```

#### **7c. Total Loss**
```python
total_loss = waypoint_loss + language_loss
# Result: 2.49
```

---

### **Step 8: Backpropagation**
📁 `models/driving.py` → `training_step()` (line 269)

**What happens:** Compute gradients and update **only LoRA weights**
```python
# PyTorch Lightning automatically calls:
loss.backward()  # Compute gradients

# Gradients flow through:
# ✅ Language model LoRA weights  → UPDATED
# ❌ Vision encoder weights        → FROZEN (no gradients)
# ❌ Language model base weights   → FROZEN (LoRA only)
```

---

### **Step 9: Optimizer Updates Weights**
📁 `models/driving.py` → `configure_optimizers()` (line 718-732)

**What happens:** AdamW optimizer updates LoRA parameters
```python
# Before update:
lora_weight_A = [[0.123, 0.456], [0.789, 0.012]]

# Apply gradient:
lora_weight_A -= learning_rate * gradient
# After update:
lora_weight_A = [[0.122, 0.455], [0.788, 0.011]]  # Slightly changed
```

**Key:** Only ~327M LoRA parameters updated, not the full 957M model.

---

## Visual Example: One Training Sample

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT                                                           │
├─────────────────────────────────────────────────────────────────┤
│ Image: [Road scene with car ahead]                             │
│ Speed: 5.2 m/s                                                  │
│ Target: [10.5, 2.3]                                            │
│ Prompt: "You are a driving assistant. Target waypoint:         │
│          <TARGET_POINT><TARGET_POINT>. Describe the scene."    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PROCESSING                                                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Vision Encoder (FROZEN): Image → Visual features            │
│    [1024×512 pixels] → [256 patches × 2048 dims]              │
│                                                                 │
│ 2. Text Embeddings: Prompt → Text features                     │
│    "You are..." → [150 tokens × 2048 dims]                    │
│                                                                 │
│ 3. Concatenate: [256 + 150 = 406 tokens × 2048 dims]          │
│                                                                 │
│ 4. Language Model (LoRA): Process combined features            │
│    [406 × 2048] → [406 × 2048 features] + [406 × vocab logits]│
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT (Predictions)                                            │
├─────────────────────────────────────────────────────────────────┤
│ Waypoints: [[1.1, 0.05], [2.2, 0.1], [3.1, 0.15], ...]       │
│ Text: "Driving straight on the road. Waypoints: [[1.1, 0.05],│
│        [2.2, 0.1], ...]"                                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ GROUND TRUTH (Labels)                                           │
├─────────────────────────────────────────────────────────────────┤
│ Waypoints: [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], ...]         │
│ Text: "Driving straight on the road. Waypoints: [[1.0, 0.0], │
│        [2.0, 0.0], ...]"                                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ LOSS COMPUTATION                                                │
├─────────────────────────────────────────────────────────────────┤
│ Waypoint Loss = |predicted - ground_truth| = 0.15              │
│ Language Loss = cross_entropy(pred_tokens, true_tokens) = 2.34 │
│ Total Loss = 0.15 + 2.34 = 2.49                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ BACKPROPAGATION                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Compute gradients: ∂Loss/∂(LoRA weights)                       │
│ Update LoRA weights: weight -= learning_rate × gradient        │
│                                                                 │
│ ✅ Updated: 327M LoRA parameters in language model             │
│ ❌ Frozen: 629M parameters (vision encoder + base LM)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Images ARE used** - Vision encoder extracts features (but stays frozen)
2. **Language IS learned** - Model learns to map visual features → text descriptions
3. **Only LoRA weights update** - Efficient fine-tuning (327M params instead of 957M)
4. **Two losses combined** - Waypoint loss (driving) + Language loss (descriptions)
5. **Supervised learning** - Model imitates expert demonstrations from dataset

---

## File Reference

| Step | File | Function | Line |
|------|------|----------|------|
| 1. Load data | `dataloader/dataset_driving.py` | `__getitem__()` | 33-319 |
| 2. Prompt → embeddings | `models/adaptors/adaptors.py` | `LanguageAdaptor.forward()` | 238-257 |
| 3. Vision features | `models/driving.py` | `forward_model()` | 200-205 |
| 4. Combine features | `models/driving.py` | `forward_model()` | 208-214 |
| 5. Language model | `models/driving.py` | `forward_model()` | 215-233 |
| 6. Extract predictions | `models/adaptors/adaptors.py` | `get_predictions()` | 163-180 |
| 7. Compute losses | `models/adaptors/adaptors.py` | `compute_loss()` | 200-274 |
| 8. Training step | `models/driving.py` | `training_step()` | 263-271 |
| 9. Optimizer | `models/driving.py` | `configure_optimizers()` | 718-732 |

