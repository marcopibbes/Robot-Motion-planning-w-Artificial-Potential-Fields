# APF with Moving Obstacles — Standard vs. Velocity-Aware

A 2D simulation comparing two variants of the **Artificial Potential Fields (APF)** path-planning algorithm in environments with **moving rectangular obstacles**. The robot navigates from a fixed start position to a goal, reacting in real time to obstacle positions (and, in the second variant, to their velocities).

![demo](demo.gif)
> *Replace `demo.gif` with your own recorded output (see [Saving a GIF](#saving-a-gif)).*

---

## Table of Contents

- [Overview](#overview)
- [Algorithm Details](#algorithm-details)
  - [Standard APF](#standard-apf)
  - [Velocity-Aware APF](#velocity-aware-apf)
- [Environment](#environment)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Saving a GIF](#saving-a-gif)
- [Project Structure](#project-structure)

---

## Overview

Artificial Potential Fields is a reactive path-planning method originally proposed by Khatib (1986). The robot is treated as a particle moving under the influence of two virtual forces:

- an **attractive force** pulling it toward the goal
- a **repulsive force** pushing it away from obstacles

This project implements and visually compares two variants **side by side** on the same randomised scenario:

| | Standard APF | Velocity-Aware APF |
|---|---|---|
| Repulsive force depends on | distance only | distance **and** relative velocity |
| Stationary obstacle behaviour | normal repulsion | identical to standard (factor = 1) |
| Approaching obstacle behaviour | normal repulsion | amplified repulsion |

---

## Algorithm Details

### Standard APF

The attractive force toward the goal is:

```
F_att = K_att * (goal - pos) / ||goal - pos||
```

The repulsive force from each obstacle is active within a radius `d_influence` and is computed from the **closest point on the obstacle surface** to the robot:

```
F_rep = K_rep * (1/d - 1/d_influence) / d²  *  (pos - closest_point) / d
```

where `d` is the distance from the robot to the closest point on the obstacle.

The total force drives the robot at each time step with a capped speed `MAX_SPEED`.

### Velocity-Aware APF

The repulsive force is multiplied by a dynamic factor that depends on the **approach component** of the relative velocity between the obstacle and the robot:

```
v_rel      = v_obstacle - v_robot
approach   = dot(v_rel, -d_hat)          # positive when obstacle is closing in
factor     = 1 + max(0, approach)

F_rep_new  = factor * F_rep_standard
```

Key properties:
- When the obstacle is **stationary** → `v_rel = 0` → `factor = 1` → behaviour is **identical to standard APF**.
- When the obstacle is **approaching** the robot → `factor > 1` → the robot reacts earlier and more aggressively.
- When the obstacle is **moving away** → `approach < 0` → `factor = 1` → no penalty.

---

## Environment

- **World size:** 20 × 20 units
- **Start:** (1, 1) — bottom-left area
- **Goal:** (18, 18) — top-right area
- **Obstacles:** 3–7 randomised rectangles per run
  - Random position (guaranteed clear of start/goal)
  - Random size (width and height independently in [1.2, 3.2])
  - Random velocity and direction; at least one obstacle is always **stationary**
  - Obstacles bounce off world boundaries
- Both APF variants are run on **the same obstacle set** (same seed) for a fair comparison

---

## Requirements

- Python ≥ 3.9
- `numpy`
- `matplotlib`
- `pillow` *(only needed to export GIFs)*

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy matplotlib pillow
```

---

## Usage

**Random scenario** (different every run):
```bash
python apf_simulation.py
```

**Reproducible scenario** with a fixed seed:
```bash
python apf_simulation.py --seed 42
```

The simulation opens an animated window with two panels running simultaneously — **APF Standard** on the left and **APF + Velocity** on the right.

The terminal prints a summary of the generated scenario:
```
Seed = 42  |  3 obstacles
  O1: pos=(9.3, 6.1)   2.8x1.9  vel=(1.23, -0.87)
  O2: pos=(13.5, 14.2) 1.5x2.4  vel=(-0.55, 1.10)
  O3: pos=(6.8, 11.7)  2.1x2.1  STATIONARY
  [standard] 381 steps  |  final pos=[17.81 17.78]
  [velocity] 385 steps  |  final pos=[17.89 17.74]
```

### Legend (in the animation)

| Symbol | Meaning |
|--------|---------|
| ● green | Start position |
| ✦ gold | Goal position |
| ■ red label | Stationary obstacle |
| ▶ white label | Moving obstacle |
| cyan trail | Robot path — Standard APF |
| orange trail | Robot path — Velocity-Aware APF |

---

## Parameters

All tunable parameters are at the top of `apf_simulation.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WORLD_SIZE` | 20.0 | Side length of the square world |
| `DT` | 0.05 | Simulation time step |
| `K_ATT` | 1.5 | Attractive force gain |
| `K_REP` | 80.0 | Repulsive force gain |
| `D_INFLUENCE` | 3.5 | Obstacle influence radius |
| `D_GOAL` | 0.3 | Goal-reached threshold |
| `MAX_SPEED` | 2.0 | Robot maximum speed |
| `MAX_STEPS` | 2000 | Maximum simulation steps per run |

---

## Saving a GIF

Requires `pillow` (included in the install step above).

```bash
# Save with a random scenario
python apf_simulation.py --gif output.gif

# Save with a reproducible scenario
python apf_simulation.py --seed 42 --gif apf_seed42.gif
```

The GIF is saved in the current working directory. Rendering may take a minute or two depending on the number of simulation steps.

---

## Project Structure

```
.
├── apf_simulation.py   # Main simulation script
├── README.md           # This file
└── demo.gif            # (optional) example animation for the README
```

---

## References

- Khatib, O. (1986). *Real-time obstacle avoidance for manipulators and mobile robots.* The International Journal of Robotics Research, 5(1), 90–98.
- Latombe, J.-C. (1991). *Robot Motion Planning.* Springer.