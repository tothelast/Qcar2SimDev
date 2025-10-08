# How It Works - Self-Driving Car Project Explained Simply

**Last Updated:** October 8, 2025

---

## Introduction

This project teaches a small robot car to drive itself through a virtual city using artificial intelligence. The car is about the size of a shoebox (1/10th the size of a real car), and it drives around in a computer-simulated world called QLabs. Just like a human driver, the car uses a camera to see the road ahead, an AI "brain" to decide where to go, and controls to steer and adjust its speed. The goal is to follow a planned route from start to finish without crashing into walls or getting lost.

---

## The Big Picture

Imagine you're driving a car. You look at the road ahead, decide where you want to go, turn the steering wheel, and press the gas or brake pedal. This project does the same thing, but with a computer controlling everything:

1. **Camera (Eyes):** A camera on the car takes pictures of what's ahead, just like your eyes see the road
2. **AI Brain (Decision Maker):** An AI model called SimLingo looks at the pictures and decides where the car should go next
3. **Control System (Hands and Feet):** A control system translates the AI's decisions into actual steering and speed commands, like your hands turning the wheel and your foot on the pedals
4. **Virtual World (Road):** The car drives in a simulated city environment where we can test it safely

The car does this entire process about 2 times per second, constantly looking ahead and adjusting its path.

---

## Step-by-Step: What Happens Each Moment

Let's walk through exactly what happens during one moment of driving:

### Step 1: The Camera Takes a Picture (0.1 seconds)

The car has a camera mounted on top that acts like its eyes. This camera:
- Takes pictures that are 1024 pixels wide and 512 pixels tall (about the shape of a wide-screen TV)
- Can see 110 degrees in front of the car (imagine spreading your arms wide - that's about how much the camera sees)
- Captures what's ahead: the road, buildings, walls, and the path to follow

The camera takes a new picture about 10 times per second, giving the AI fresh information constantly.

### Step 2: The Picture Gets Prepared for the AI (0.05 seconds)

Before the AI can use the picture, it needs to be prepared (preprocessed):

1. **Crop the bottom:** The bottom 30% of the image is removed because it just shows the car's hood, which isn't useful
2. **Split into patches:** The remaining image is split into 2 square pieces, each 512×512 pixels. Think of it like cutting a wide photo into two overlapping squares
3. **Normalize colors:** The colors are adjusted to match what the AI was trained on, like adjusting the brightness and contrast on your TV

### Step 3: The AI Brain Predicts Where to Go (0.5 seconds)

Now comes the interesting part. The AI model (called SimLingo) receives:
- The 2 image patches showing what's ahead
- The car's current speed (in meters per second)

The AI model is like a very experienced driver who has driven thousands of miles in a similar environment. It looks at the images and thinks: "Based on what I see and how fast I'm going, here's where I should drive next."

The AI outputs two sets of predictions:

**Route Waypoints (20 points):**
- These are 20 target points that form a path ahead of the car
- Each point has an X and Y coordinate telling the car where to aim
- Think of them like breadcrumbs showing the path forward
- They're spaced about 0.1 to 0.5 meters apart (a few inches to a couple feet)
- These points are in "ego frame" - meaning they're described relative to where the car is now (like saying "2 feet forward and 1 foot to the left" instead of using GPS coordinates)

**Speed Waypoints (10 points):**
- These are 10 additional points that help the car figure out how fast it should be going
- The AI uses these to suggest whether the car should speed up, slow down, or maintain speed
- For example, if there's a sharp turn ahead, the speed waypoints will be closer together, signaling the car to slow down

All these waypoints together form a matrix (a grid of numbers) that's 30 rows by 2 columns - 30 points total, each with an X and Y position.

### Step 4: Converting Predictions to Steering and Speed (0.05 seconds)

The AI's waypoint predictions need to be converted into actual commands the car can follow. This is where the control system comes in - think of it as the part that actually moves your hands and feet when driving.

**Steering Control (Lateral PID Controller):**

The steering controller looks at the route waypoints and figures out which direction to turn the steering wheel. Here's how it works:

1. **Pick a target point:** Based on the car's current speed, it picks one of the 20 waypoints to aim for. If the car is going slow (less than 5.5 m/s, which is about 12 mph), it looks at a point 2.25 meters ahead (about 7 feet). If going faster, it looks further ahead - up to 7 meters (23 feet).

2. **Calculate the error:** It measures how far off the car is from pointing directly at that target point. This is called the "heading error."

3. **Apply PID control:** This is a mathematical formula (don't worry about the details) that smoothly adjusts the steering based on:
   - How far off you are right now (Proportional)
   - How far off you've been over the past few moments (Integral)
   - How quickly the error is changing (Derivative)
   
   Think of it like steering a bicycle: you turn more sharply if you're way off course, and you make smaller adjustments if you're almost on track.

The steering output is a number between -1 and 1, where -1 means "turn fully left," 1 means "turn fully right," and 0 means "go straight."

**Speed Control (Longitudinal Linear Regression Controller):**

The speed controller looks at the speed waypoints and decides how much to press the gas pedal (throttle). Here's the process:

1. **Calculate desired speed:** It looks at how far apart the speed waypoints are. If they're spread out, it means the car should go faster. If they're close together, it should slow down. Specifically, it measures the distance between waypoint #3 and waypoint #8, then multiplies by 2 to get the target speed.

2. **Compare to current speed:** It checks how fast the car is actually going versus how fast it should be going.

3. **Calculate throttle:** Using a formula based on 7 pre-calculated numbers (coefficients), it determines how much throttle to apply. This formula was learned from thousands of examples of good driving.

The throttle output is a number between 0 and 1, where 0 means "no gas" and 1 means "full throttle."

### Step 5: The Car Moves (0.1 seconds)

Finally, the steering and throttle commands are sent to the car:
- The steering command turns the front wheels left or right
- The throttle command makes the car speed up or slow down
- The car moves forward based on these commands

Then the whole process repeats - camera takes a new picture, AI makes new predictions, controls adjust, and the car keeps driving.

---

## Key Components Explained

### The Virtual Car (QCar2)

The car in this project is a virtual version of a real robot car called QCar2:
- **Size:** 1/10th scale - about 40 cm (16 inches) long, like a large toy car
- **Speed:** Typically drives at 0.7 meters per second (about 1.5 mph, a slow walking pace)
- **Maximum speed:** Can reach 1.27 meters per second (about 2.8 mph, a brisk walk)
- **Acceleration:** Much slower than a real car - it takes time to speed up or slow down

### The Virtual World (QLabs Cityscape)

The car drives in a computer-simulated city environment:
- **Size:** About 100 meters by 100 meters (roughly the size of a city block)
- **Features:** Roads, buildings, walls, intersections, and even a roundabout
- **Coordinates:** Every location has an X, Y, and Z coordinate (like latitude, longitude, and altitude)
- **Physics:** The simulation includes realistic physics - the car can collide with walls, has momentum, and responds to steering

### The AI Brain (SimLingo Model)

SimLingo is a type of AI called a "vision-language-action model":

**What it is:**
- A neural network with 1 billion parameters (think of parameters as tiny adjustable knobs that were tuned during training)
- Based on a model called InternVL2-1B, which was originally designed to understand images and text
- Fine-tuned specifically for driving using a technique called LoRA (Low-Rank Adaptation)

**How it was trained:**
- Trained on thousands of hours of driving data from a different simulator called CARLA
- CARLA simulates full-size cars driving in realistic city environments
- The AI learned by watching examples of good driving: "When you see this road scene, you should steer this way"

**What makes it special:**
- It can look at a camera image and predict a safe path forward
- It doesn't need detailed maps or GPS - it just uses what it sees
- It learned to handle different road conditions: straight roads, curves, intersections, roundabouts

**Size and speed:**
- The model file is about 1.8 GB (like a high-definition movie)
- Takes about 0.5 seconds to process one image on a powerful graphics card (GPU)
- This is why the car only updates its decisions 2 times per second instead of faster

### The Route (Planned Path)

Before the car starts driving, we give it a planned route:
- **36 waypoints** marking the path from start to finish
- **Total distance:** About 90 meters (roughly the length of a football field)
- **Includes:** Straight sections, curves, and a roundabout
- **Start point:** [2.686, 18.498, 0.005] in world coordinates
- **End point:** [-19.841, 29.760, 0.0] in world coordinates

Think of this route like the blue line on Google Maps showing you where to drive.

---

## How the Car Knows Where to Go

### Position Tracking (Localization)

The car always knows exactly where it is in the virtual world:
- The simulation provides the car's precise X, Y, Z coordinates
- It also knows which direction it's facing (heading angle)
- This is like having a perfect GPS that updates 10 times per second

### Understanding "Ego Frame" vs. "World Frame"

This is an important concept that can be confusing, so let's break it down:

**World Frame (Global Coordinates):**
- Fixed coordinates that never change
- Like addresses on a map: "123 Main Street" is always in the same place
- The route waypoints are stored in world frame
- Example: The start point is at [2.686, 18.498] - this location never changes

**Ego Frame (Car-Centric Coordinates):**
- Coordinates relative to where the car is right now
- Like giving directions: "Go 2 meters forward, then 1 meter to your left"
- The AI's predictions are in ego frame
- Example: A waypoint at [2.0, 0.5] means "2 meters ahead of you and 0.5 meters to your left"

**Why this matters:**
- The AI predicts waypoints in ego frame because it's easier to learn: "Based on what I see ahead, I should aim 2 meters forward and slightly left"
- But the route is stored in world frame so we can track progress: "The car should be at position [5.0, 20.0] right now"
- The system constantly converts between these two frames

### Following the Path

Here's how the car follows the planned route:

1. **Find current position:** The car knows where it is in world coordinates
2. **Find nearest waypoint:** It looks at the route and finds which waypoint it's closest to (like finding your current position on a map)
3. **Convert route to ego frame:** It takes the upcoming waypoints from the route and converts them to ego frame (relative to where the car is now)
4. **Compare with AI predictions:** The AI also predicts waypoints in ego frame
5. **Use AI predictions for control:** The steering and speed controllers use the AI's predicted waypoints to decide how to move
6. **Track progress:** The system checks if the car is staying close to the planned route

### Measuring Success

We measure how well the car is doing using several metrics:

**Success Rate (27.7%):**
- This measures what percentage of the time the car is within 1 meter (about 3 feet) of the planned route
- Currently, the car is within 1 meter of the route about 28% of the time
- The other 72% of the time, it's further away (but still generally following the route)

**Lateral Deviation (1.55 meters average):**
- This measures how far sideways the car is from the planned route
- On average, the car is 1.55 meters (about 5 feet) away from where it should be
- Sometimes it's right on track (0 meters), sometimes it's further off (up to 2.6 meters)

**Collision Rate (4.1%):**
- About 4% of the time, the car touches a wall or obstacle
- These are usually brief bumps, not major crashes
- The car typically recovers and continues driving

---

## Challenges: Why This Is Difficult

### The Domain Gap Problem

Imagine you learned to drive in a full-size sedan on highways, and then someone asked you to drive a go-kart on a miniature race track. Everything would feel different:
- The go-kart is much smaller and lighter
- It accelerates and turns differently
- The track is tighter with sharper curves
- Your instincts from driving the sedan don't perfectly transfer

This is exactly the challenge our AI faces:

**Training Environment (CARLA):**
- Full-size cars (like sedans and SUVs)
- Fast acceleration: 3-5 meters per second squared
- Typical speeds: 5-15 meters per second (11-34 mph)
- Wide roads and gentle curves

**Deployment Environment (QLabs QCar2):**
- Tiny car (1/10th scale)
- Slow acceleration: 0.56 meters per second squared (about 10 times slower!)
- Typical speeds: 0.5-1.5 meters per second (1-3 mph)
- Tighter spaces and sharper turns

The AI learned to drive in CARLA, but now it's driving in QLabs. Its predictions aren't perfectly calibrated for the smaller, slower car. This is called the "domain gap."

### Why the Success Rate Isn't Higher

With a 27.7% success rate, you might wonder: "Why isn't the car doing better?" Here are the main reasons:

1. **Domain Gap:** As explained above, the AI's training doesn't perfectly match the deployment environment

2. **Different Dynamics:** The QCar2 responds differently to steering and throttle commands than the cars in CARLA. When the AI says "steer this much," the QCar2 might turn more or less than expected.

3. **Speed Limitations:** The car is much slower than what the AI was trained on. The AI might predict a path that assumes the car can accelerate quickly, but the QCar2 can't keep up.

4. **Tight Spaces:** The QLabs environment has some tight curves and a roundabout. The AI sometimes predicts paths that are too wide for these tight spaces.

5. **Inference Speed:** The AI takes 0.5 seconds to process each image. In that time, the car has already moved about 35 cm (14 inches). This delay means the AI is always slightly behind what's actually happening.

### What "Good Enough" Looks Like

Despite these challenges, the car performs reasonably well:
- ✅ It completes the entire route (reaches the destination)
- ✅ It drives at a reasonable speed (0.7 m/s average)
- ✅ It only collides with walls 4% of the time
- ✅ It generally follows the planned path (even if not perfectly)

For a research project testing an AI trained in one environment and deployed in another, this is actually quite good! It shows that the AI has learned general driving skills that transfer across different environments.

---

## Current Performance Summary

Let's put all the numbers in context:

**What the car does well:**
- **Route completion:** Successfully drives from start to finish (95.7 meters)
- **Speed:** Maintains a steady pace (0.7 m/s average, 1.27 m/s maximum)
- **Throttle control:** Smoothly accelerates and maintains speed
- **Recovery:** When it deviates from the path, it usually corrects itself

**What could be improved:**
- **Lateral tracking:** Often drifts 1-2 meters away from the ideal path
- **Tight curves:** Has more difficulty with sharp turns and the roundabout
- **Success rate:** Only within 1 meter of the route 28% of the time

**Why we're not changing the control parameters:**
- We tried adjusting the steering controller to reduce oscillations
- The custom tuning actually made things worse (success rate dropped to 15%)
- The official parameters from the original SimLingo research work best
- This suggests the issue is the domain gap, not the control system

---

## What's Next?

To improve the car's performance, here are the planned next steps:

**Short-term:**
- Continue testing and collecting data
- Monitor performance over multiple runs
- Document patterns in where the car struggles

**Medium-term:**
- Collect driving data specifically in the QLabs environment
- Fine-tune the AI model using this QLabs data
- This should help bridge the domain gap

**Long-term:**
- Explore hybrid approaches (combining AI with traditional control methods)
- Train the AI on data from both CARLA and QLabs
- Investigate domain adaptation techniques

---

## Conclusion

This project demonstrates how an AI model can learn to drive a car by looking at camera images, even when deployed in an environment different from where it was trained. While the car doesn't drive perfectly (28% success rate), it successfully completes routes and shows that learned driving skills can transfer across different simulators and vehicle types.

The main challenge is the domain gap - the difference between the full-size cars in CARLA where the AI was trained, and the small QCar2 in QLabs where it's deployed. Despite this challenge, the car drives reasonably well, completing routes at a steady pace with only occasional collisions.

This is an active research project, and future improvements will focus on fine-tuning the AI specifically for the QLabs environment to improve its performance.

---

## Glossary of Technical Terms

- **AI Model / SimLingo:** The "brain" that looks at images and predicts where to drive
- **Camera / CSI Camera:** The "eyes" that capture images of the road ahead
- **Waypoints:** Target points that form a path for the car to follow
- **Ego Frame:** Coordinates relative to the car's current position ("2 meters ahead")
- **World Frame:** Fixed coordinates in the virtual world (like GPS coordinates)
- **PID Controller:** A mathematical formula that smoothly adjusts steering based on errors
- **Linear Regression Controller:** A formula that calculates how much throttle to apply
- **Lateral Deviation:** How far sideways the car is from the planned route
- **Success Rate:** Percentage of time the car is within 1 meter of the route
- **Domain Gap:** The difference between the training environment and deployment environment
- **QLabs:** The virtual city environment where the car drives
- **QCar2:** The small robot car (1/10th scale)
- **CARLA:** The simulator where the AI was originally trained
- **Localization:** Knowing the car's exact position in the world
- **Inference:** The process of the AI making predictions from an image
- **Throttle:** The "gas pedal" - how much power to apply
- **Steering:** Turning the wheels left or right
- **Collision:** When the car touches a wall or obstacle


