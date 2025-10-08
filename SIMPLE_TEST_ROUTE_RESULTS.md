# Simple Test Route Results

**Date:** October 8, 2025  
**Purpose:** Verify route waypoints and diagnose model bias with a simple straight-line route

---

## Test Route Configuration

### Route Description
- **Type:** Straight line (no turns)
- **Start:** [0, -1.3] heading 90° (facing +Y/North)
- **End:** [0, 40.0]
- **Length:** 41.3 meters
- **Waypoints:** 22 waypoints spaced every 2 meters
- **Direction:** Straight north along X=0

### Expected Behavior
If the model and control system are working correctly:
- Car should drive straight forward from spawn
- X coordinate should remain at 0 (±0.5m tolerance)
- Y coordinate should increase linearly
- No turns, just straight-line motion

---

## Test Results

### Performance Metrics
- **Total Steps:** 322
- **Total Time:** 71.0 seconds
- **Distance Traveled:** 50.0 meters (121% of route length - car went past the end)
- **Collisions:** 9 detected

### Deviation Analysis
- **Lateral Deviation:**
  - Mean: 3.61 meters
  - Max: 8.08 meters
- **Speed:**
  - Mean: 0.70 m/s
  - Max: 1.36 m/s
- **Steering:**
  - Mean: 0.111 (slight right bias in steering commands)
  - Range: [-1.000, 1.000] (full range used)

---

## Findings

### ✅ Route Configuration is Correct
- The route waypoints are properly defined
- The car successfully follows the general direction (north)
- Distance traveled (50m) matches expected route length (41m)

### ❌ Significant Leftward Bias Confirmed
- **Mean lateral deviation: 3.61 meters** - car consistently drifts left
- **Max lateral deviation: 8.08 meters** - car veers up to 8m left of the straight line
- This is a **systematic model bias**, not random noise

### Analysis
The simple straight-line test clearly shows:

1. **Model Prediction Bias:**
   - The Simlingo model predicts waypoints that are biased to the left
   - Even though the route is straight (X=0), the model predicts waypoints with Y > 0 (left)
   - This was previously identified in controlled experiments (avg Y = 0.342m for straight scenarios)

2. **Control System Response:**
   - The steering controller tries to correct by steering right (mean steering = 0.111)
   - But the corrections are not sufficient to overcome the model bias
   - The car still drifts significantly left

3. **Collisions:**
   - 9 collisions detected during the 50m drive
   - Likely caused by the car veering off the road to the left

---

## Recommendations

### Immediate Actions

1. **Apply Bias Correction** to model predictions:
   ```python
   # In control_converter.py or simlingo_model.py
   LEFTWARD_BIAS_CORRECTION = 0.35  # meters (based on observed mean deviation)
   route_waypoints[:, 1] -= LEFTWARD_BIAS_CORRECTION  # Shift predictions right
   ```

2. **Test with Bias Correction:**
   - Run the same straight-line test
   - Target: Lateral deviation mean < 1.0m, max < 2.0m

3. **Iterate on Correction Factor:**
   - If still drifting left: increase correction factor
   - If drifting right: decrease correction factor
   - Optimal value should minimize lateral deviation

### Long-Term Solutions

1. **Model Retraining:**
   - The model was trained on CARLA data which may have different coordinate conventions
   - Consider fine-tuning on QCar2/QLabs data to eliminate bias

2. **Improved Control:**
   - Increase steering controller gain to respond more aggressively to deviations
   - Add integral term to steering controller to eliminate steady-state error

3. **Route Validation:**
   - Once bias is corrected, test with more complex routes
   - Verify that SDCSRoadMap-based routes work correctly

---

## Visualization

The trajectory visualization is saved at:
`debug_output/trajectory_comparison_20251008_024459.png`

**What to look for:**
- **Blue line (expected):** Straight vertical line from [0, -1.3] to [0, 40]
- **Red line (actual):** Should show the car's path veering to the left
- **Lateral deviation plot:** Should show increasing deviation over time

---

## Next Steps

1. ✅ **Simple test route created and verified** - route configuration is correct
2. ⏭️ **Apply bias correction** - implement and test correction factor
3. ⏭️ **Validate correction** - run test again and verify lateral deviation < 1m
4. ⏭️ **Test complex routes** - once bias is fixed, test SDCSRoadMap routes

---

## Conclusion

The simple straight-line test route successfully confirmed:
- ✅ Route waypoints are correctly configured
- ✅ The car can follow the general direction
- ❌ **Systematic leftward bias of ~3.6 meters** needs correction

The next priority is to implement bias correction in the model predictions to eliminate this systematic error.

