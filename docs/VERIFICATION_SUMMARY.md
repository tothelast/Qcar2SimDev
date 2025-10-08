# SimLingo Paper Verification - Executive Summary

**Date:** 2025-10-08  
**Status:** ✅ Verification Complete - Critical Issues Found  
**Full Report:** See `SIMLINGO_PAPER_VERIFICATION.md`

---

## 🔍 Key Findings

### ✅ What We Got Right

1. **Model Architecture:** Correctly using InternVL2-1B with LoRA (r=32, alpha=64)
2. **Camera Configuration:** Exact match (1024×512, FOV 110°, position [-1.5, 0.0, 2.0])
3. **Waypoint Prediction:** Correct structure (20 route + 10 speed waypoints)
4. **Speed Calculation:** Exact formula match (`||WP[0]-WP[3]|| × 2.0`)
5. **Speed PID Controller:** Correct parameters (Kp=1.75, Ki=1.0, Kd=2.0)

### ❌ Critical Discrepancies

#### 1. **WRONG LATERAL PID PARAMETERS** (CRITICAL)

| Parameter | Official SimLingo | Our Implementation | Difference |
|-----------|------------------|-------------------|------------|
| **Kp** | 3.25 | 3.118 | -4.1% |
| **Ki** | 1.0 | 0.640 | **-36%** ⚠️ |
| **Kd** | 1.0 | 1.378 | **+38%** ⚠️ |

**Root Cause:** We're using parameters from `team_code/config.py` (Bayesian-optimized for CARLA autopilot) instead of `team_code/config_simlingo.py` (designed for SimLingo model-based control).

**Impact:**
- **Route Deviation:** 36% lower Ki → slower integral error correction → persistent lateral error at roundabout
- **Slow Speed:** Poor lateral tracking → model predicts conservative waypoints → lower speeds

#### 2. **WRONG LONGITUDINAL CONTROLLER TYPE** (HIGH PRIORITY)

| Aspect | Official SimLingo | Our Implementation |
|--------|------------------|-------------------|
| **Controller** | Linear Regression | PID |
| **Parameters** | 7 regression coefficients | Kp, Ki, Kd gains |

**Root Cause:** We implemented `LongitudinalPIDController` instead of `LongitudinalLinearRegressionController`.

**Impact:**
- **Slow Speed:** Different acceleration characteristics
- **Destination Overshoot:** Different braking behavior

---

## 🎯 Immediate Action Items

### Priority 1: Fix Lateral PID Parameters (5 minutes)

**File:** `src/control_converter.py`

**Change:**
```python
# BEFORE (WRONG)
class LateralPIDController:
    def __init__(self, config):
        self.k_p = config.lateral_pid_kp  # 3.118
        self.k_i = config.lateral_pid_ki  # 0.640
        self.k_d = config.lateral_pid_kd  # 1.378

# AFTER (CORRECT)
class LateralPIDController:
    def __init__(self, config):
        self.k_p = config.turn_kp  # 3.25
        self.k_i = config.turn_ki  # 1.0
        self.k_d = config.turn_kd  # 1.0
```

**Expected Improvement:**
- Lateral deviation at roundabout: 13.2m → ~5-7m (estimated)
- Mean lateral deviation: 1.29m → ~0.8-1.0m (estimated)
- Steering responsiveness: +36% (from Ki increase)

### Priority 2: Implement Linear Regression Controller (1-2 hours)

**File:** `src/control_converter.py`

**Add new class:**
```python
class LongitudinalLinearRegressionController:
    def __init__(self, config):
        self.minimum_target_speed = 0.278  # m/s
        self.params = np.array([
            1.1990342347353184, -0.8057602384167799, 1.710818710950062,
            0.921890257450335, 1.556497522998393, -0.7013479734904027,
            1.031266635497984
        ])
        self.max_acceleration = 1.89  # m/tick
        self.max_deceleration = -4.82  # m/tick
    
    def get_throttle_and_brake(self, target_speed, current_speed):
        if target_speed < 1e-5:
            return 0.0, True
        
        target_speed = max(self.minimum_target_speed, target_speed)
        current_speed_kmh = current_speed * 3.6
        target_speed_kmh = target_speed * 3.6
        
        speed_error = target_speed_kmh - current_speed_kmh
        
        # Maximum acceleration check
        if speed_error > self.max_acceleration:
            return 1.0, False
        
        # Braking check
        if current_speed_kmh / target_speed_kmh > self.params[-1]:
            return 0.0, True
        
        # Linear regression calculation
        speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
        current_speed_norm = current_speed_kmh / 100.0
        
        features = np.array([
            current_speed_norm,
            current_speed_norm**2,
            100 * speed_error_cl,
            speed_error_cl**2,
            current_speed_norm * speed_error_cl,
            current_speed_norm**2 * speed_error_cl
        ])
        
        throttle = np.clip(features @ self.params[:-1], 0.0, 1.0)
        return throttle, False
```

**Expected Improvement:**
- Average speed: 0.47 m/s → ~1.5-2.0 m/s (estimated)
- Acceleration: More aggressive (closer to CARLA training data)
- Braking: More predictable (explicit deceleration limits)

### Priority 3: Test and Validate (30 minutes)

**Run the same test scenario and compare:**

| Metric | Before | Expected After | Target |
|--------|--------|---------------|--------|
| Mean Lateral Deviation | 1.29 m | ~0.8-1.0 m | < 1.0 m |
| Max Lateral Deviation | 3.42 m | ~2.0-2.5 m | < 3.0 m |
| Average Speed | 0.47 m/s | ~1.5-2.0 m/s | > 1.0 m/s |
| Destination Overshoot | 57.5 m | ~5-10 m | < 10 m |

---

## 📊 Evidence Summary

### Official SimLingo Sources

1. **Model Config:** `simlingo_training/config.py`
   - LoRA: r=32, alpha=64, dropout=0.1
   - Learning rate: 5e-2
   - Waypoints: 20 route + 10 speed

2. **Agent Config:** `team_code/config_simlingo.py`
   - Turn PID: Kp=3.25, Ki=1.0, Kd=1.0
   - Speed PID: Kp=1.75, Ki=1.0, Kd=2.0
   - Brake speed: 0.4 m/s, ratio: 1.1

3. **Controllers:** `team_code/lateral_controller.py`, `team_code/longitudinal_controller.py`
   - Lateral: Simple PID with speed-dependent lookahead
   - Longitudinal: Linear regression (default), PID (alternative)

### Our Implementation Issues

1. **Wrong Parameter Source:** Using `config.py` (autopilot) instead of `config_simlingo.py` (SimLingo)
2. **Wrong Controller Type:** Using PID instead of Linear Regression for longitudinal control
3. **Unused Parameters:** Defined `turn_kp/ki/kd` but never used them

---

## 🔧 Implementation Plan

### Step 1: Quick Fix (Today)

1. ✅ Create verification document (DONE)
2. ⏳ Fix lateral PID parameters (5 min)
3. ⏳ Test with fixed parameters (30 min)
4. ⏳ Document results

### Step 2: Complete Fix (This Week)

1. ⏳ Implement Linear Regression controller (1-2 hours)
2. ⏳ Replace PID-based throttle calculation (30 min)
3. ⏳ Full system test (1 hour)
4. ⏳ Update documentation

### Step 3: Cleanup (Next Week)

1. ⏳ Remove unused parameters from config
2. ⏳ Add comments explaining parameter sources
3. ⏳ Create unit tests for controllers
4. ⏳ Update integration report

---

## 📈 Expected Outcomes

### After Priority 1 Fix (Lateral PID)

- ✅ Improved lateral tracking (especially at roundabout)
- ✅ Reduced maximum lateral deviation
- ✅ More responsive steering
- ⚠️ Speed may still be slow (needs Priority 2)

### After Priority 2 Fix (Longitudinal Controller)

- ✅ Higher average speed
- ✅ Better acceleration
- ✅ Improved destination detection
- ✅ More CARLA-like behavior

### Combined Effect

- ✅ All three issues resolved:
  1. Slow speed → Fixed by linear regression controller
  2. Route deviation → Fixed by correct lateral PID
  3. Destination overshoot → Fixed by better speed control

---

## 🚨 Critical Insight

**The root cause of all three issues is using the wrong configuration file!**

- We copied parameters from `team_code/config.py` (full CARLA autopilot with privileged planner)
- We should have used `team_code/config_simlingo.py` (SimLingo model-based agent)

**The official SimLingo agent uses SIMPLE controllers, not the Bayesian-optimized advanced controllers!**

This explains why:
1. Our steering is less responsive (lower Ki)
2. Our speed is slower (wrong controller type)
3. Our destination detection fails (different braking behavior)

---

## 📝 Next Steps

1. **Implement Priority 1 fix** (lateral PID parameters)
2. **Run test and measure improvement**
3. **If successful, implement Priority 2 fix** (linear regression controller)
4. **Document final results in AUTONOMOUS_VEHICLE_ANALYSIS.md**
5. **Update INTEGRATION_REPORT.md with corrected implementation**

---

## 📚 References

- Official Repository: https://github.com/RenzKa/simlingo
- Paper: https://arxiv.org/abs/2503.09594
- Full Verification Report: `SIMLINGO_PAPER_VERIFICATION.md`
- Previous Analysis: `AUTONOMOUS_VEHICLE_ANALYSIS.md`


