# Action Dreaming - Instruction Following Guide

## Overview

Your SimLingo model **WAS trained with the Dreamer dataset**, which enables **instruction following** capabilities. This allows you to give natural language commands that actually affect the vehicle's driving behavior.

## What is Action Dreaming?

Action Dreaming is a training technique where the model learns multiple alternative instruction-action pairs for the same visual scene. This forces the model to:
1. **Listen to the language instruction** (not just infer from vision)
2. **Align language understanding with action space**
3. **Execute diverse instructions** including out-of-distribution commands

## Training Data

Your model was trained with:
- **50% Dreamer data** mixed with 50% expert trajectories
- **Two modes:**
  - `<INSTRUCTION_FOLLOWING>` (50%) - Execute the instruction
  - `<SAFETY>` (50%) - Reject unsafe instructions

## Instruction Types

The Dreamer dataset includes these instruction categories:

### 1. **Speed Adjustments**
- `Slow down`
- `Speed up`
- `Drive at X m/s` (target speed)

### 2. **Lane Changes**
- `Change lane to the left`
- `Change lane to the right`
- `Move to parking lane`
- `Move to sidewalk` (unsafe, for testing)

### 3. **Object-Centric**
- `Drive towards the traffic cone`
- `Drive towards the vehicle ahead`
- `Avoid the obstacle on the right`

### 4. **Road Markings**
- `Cross the stop line`
- `Ignore the stop sign` (unsafe, for testing)

## Correct Prompt Format

### **With Instruction (Dreamer Mode):**
```
<INSTRUCTION_FOLLOWING> Current speed: 5.23 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Slow down
```

### **Default (No Instruction):**
```
Current speed: 5.23 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?
```

## How to Use in GUI

1. **Enter command** in the right panel text field
2. **Press Enter** or click "Send"
3. **Observe behavior** - The model should adapt its waypoints and speed
4. **Click "Clear"** to return to default behavior

## Example Commands

### ✅ **Recommended (Trained Instructions)**

**Speed Control:**
```
Slow down
Speed up
Drive at 3 m/s
Drive at 8 m/s
```

**Lane Changes:**
```
Change lane to the left
Change lane to the right
Move to the left lane
Move to the right lane
```

**Object-Centric:**
```
Drive towards the cone
Avoid the vehicle ahead
Keep distance from the car
```

### ⚠️ **Experimental (May Work)**

```
Turn left at the intersection
Turn right at the intersection
Follow the road
Stay in lane
```

### ❌ **Won't Work (Not in Training Data)**

```
Stop                    # Not a Dreamer instruction type
Park the car            # Not trained
Do a U-turn             # Not trained
Reverse                 # Not supported
```

## Expected Behavior

### **What WILL Change:**
1. **Waypoint predictions** - Path and speed waypoints adapt to instruction
2. **Driving behavior** - Vehicle actually executes the command
3. **Commentary** - Language output reflects the instruction

### **What WON'T Change:**
- **Visual processing** - Model still sees the same scene
- **Target waypoints** - Navigation goals remain the same
- **Safety constraints** - Model may reject dangerous instructions

## Limitations

1. **Visual Context Matters** - Instructions that conflict with visual input may be ignored
   - Example: "Change lane left" when there's no left lane
   
2. **Training Distribution** - Commands similar to training data work best
   - Example: "Slow down" works better than "Stop"

3. **No Guarantees** - The model may prioritize safety over instruction
   - Example: Won't accelerate into an obstacle even if instructed

4. **Language Understanding** - Model uses InternVL2-1B, not a full LLM
   - Simple, direct commands work best

## Technical Details

### Code Implementation
- **HLC Setting:** `src/simlingo_model.py:50-68`
- **Prompt Construction:** `src/simlingo_model.py:349-359`
- **GUI Control:** `src/commentary_window.py:101-118`

### Training Statistics
- **Dreamer Data:** 50% of training samples
- **Instruction Following:** 50% of Dreamer samples
- **Safety Mode:** 50% of Dreamer samples
- **Total Epochs:** 14
- **Batch Size:** 96

### Model Architecture
- **Vision Encoder:** InternViT-300M-448px
- **Language Model:** Qwen2-0.5B-Instruct (LoRA finetuned)
- **Action Heads:** Path waypoints (20×2) + Speed waypoints (10×2)

## Troubleshooting

### Problem: Command doesn't affect behavior
**Solution:** 
1. Check if command is similar to training data
2. Verify visual context supports the instruction
3. Try simpler, more direct commands

### Problem: Model ignores instruction
**Solution:**
1. The model may be in SAFETY mode (rejecting unsafe commands)
2. Visual input may override language instruction
3. Try a different phrasing

### Problem: Waypoints change but vehicle doesn't move correctly
**Solution:**
1. This is a controller issue, not model issue
2. Check PID controller parameters
3. Verify waypoint-to-control conversion

## Success Metrics (from Paper)

The paper reports success rates for different instruction types on Town 13:

| Instruction Type | Success Rate |
|-----------------|--------------|
| Slow down       | ~70-80%      |
| Speed up        | ~70-80%      |
| Target Speed    | ~60-70%      |
| Lane Change     | ~50-60%      |
| Object-centric  | ~40-50%      |

These are **open-loop** metrics (predicted waypoints compared to ground truth), not closed-loop driving performance.

## References

- **Paper:** SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment (CVPR 2025)
- **Section:** 3.2 Datasets - Action Dreaming
- **Appendix:** A.4 Action Dreamer
- **Config:** `models/simlingo/.hydra/config.yaml` (lines 50-51, 75-76)

