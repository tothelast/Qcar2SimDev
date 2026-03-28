# Video Demo Findings (2026-03-27)

These videos are for the thesis presentation only (not the report).

## Video 1: finetuned.mp4
- **Model**: Fine-tuned (epoch_14.pt)
- **PID**: QLabs-tuned (kp=12.0, ki=0.0, kd=3.5, n=4)
- **Scene**: Roundabout navigation with obstacle variant 1

**Findings**:
- Car approached obstacle and stopped successfully
- Commentary window: "remain stopped because of the car in front" — correct description, despite no QLabs language data being collected
- Q&A window: user asked "why is the vehicle stopped?" — model responded
- Dreamer with safety ON: "move faster" → vehicle remained stationary (safety rejected the instruction)
- Dreamer with safety OFF: "move fast" → vehicle slowly moved and collided with obstacle

## Video 2: pretrained.mp4
- **Model**: Pre-trained CARLA (epoch=013.ckpt)
- **PID**: QLabs-tuned (kp=12.0, ki=0.0, kd=3.5, n=4)
- **Scene**: Roundabout navigation with obstacle variant 1

**Findings**:
- Car was stuck initially (near-zero speed, consistent with test harness result: 4.7% coverage, 1.94m in 150s)
- Commentary and Q&A windows opened — produced descriptions
- Dreamer with safety ON: "move faster" → rejected
- Dreamer with safety OFF: "move faster" → car moved forward, approached obstacle, collided with it

## Video 3: finetunedmodel_pidtuned.mp4
- **Model**: Fine-tuned (epoch_14.pt)
- **PID**: QLabs-tuned (kp=12.0, ki=0.0, kd=3.5, n=4)
- **Scene**: Baseline route (no obstacles)

**Findings**:
- Vehicle followed the route and approached the finish as expected
- Commentary window not used

## Video 4: finetunedmodel_piduntuned.mp4
- **Model**: Fine-tuned (epoch_14.pt)
- **PID**: Original SimLingo (kp=1.25, ki=0.75, kd=0.3, n=20)
- **Scene**: Baseline route (no obstacles)

**Findings**:
- Vehicle deviated from route significantly
- Collided with walls and other obstacles
- Did not reach the finish
- Commentary window not used

## Video 5: buffer_overflow.mp4
- **Model**: Fine-tuned (epoch_14.pt)
- **PID**: QLabs-tuned
- **Scene**: Roundabout navigation with 6 actors (north_pedestrian, stop_sign_roundabout, north_crosswalk, north_crosswalk_light, roundabout_car, circular_car)

**Findings**:
- Running inference with multiple dynamic actors (2 autonomous vehicles + pedestrian) on the roundabout route triggered repeated buffer overflow errors
- Error message in terminal:
  ```
  Error parsing multiple packets in receive buffer.  Clearing internal buffers.
  Error parsing multiple packets in receive buffer.  Clearing internal buffers.
  Error parsing multiple packets in receive buffer.  Clearing internal buffers.
  ```
- Reproduces the platform-level limitation documented in the report (Section 4, `sec:multi-actor-constraints`): the QLabs API serializes all control and state-query interactions, and the aggregate command rate from ego control + dynamic actor threads exceeds what the communication channel can handle
- This is why data collection and evaluation were restricted to single-obstacle scenes

## Key Takeaways

1. **Fine-tuning the driving head is necessary**: Pre-trained CARLA model cannot drive in QLabs (Video 2 vs Video 1)
2. **PID parameter tuning is necessary**: Even with the fine-tuned model, original CARLA PID values cause route deviation and collisions (Video 4 vs Video 3)
3. **Language capabilities transfer without QLabs data**: Commentary produced a correct description in the obstacle scenario despite no language data collection. Dreamer safety mechanism also transferred from CARLA training.
4. **Dreamer safety is based on memorized patterns**: Safety classification works in both models (pre-trained and fine-tuned), rejecting "move faster" when safety is ON, regardless of actual scene understanding.
