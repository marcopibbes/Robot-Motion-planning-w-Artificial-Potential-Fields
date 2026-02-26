# APF with Moving Obstacles — Standard vs. Velocity-Aware

A 2D simulation comparing two variants of the **Artificial Potential Fields (APF)** path-planning algorithm in environments with **moving rectangular obstacles**. The robot navigates from a fixed start position to a goal, reacting in real time to obstacle positions (and, in the second variant, to their velocities). Two safety mechanisms handle failure modes: a **local minimum escape** prevents permanent stalls, and an **emergency mode** guarantees collision avoidance when an obstacle comes dangerously close.

![demo](apf_result.gif)

---

## Table of Contents

- [Overview](#overview)
- [Algorithm Details](#algorithm-details)
  - [Standard APF](#standard-apf)
  - [Velocity-Aware APF](#velocity-aware-apf)
  - [Local Minimum Escape](#local-minimum-escape)
  - [Emergency Mode](#emergency-mode)
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

Both variants share the same **escape mechanism** for local minima (see below).

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

The attractive force uses a **position-dependent hybrid potential** (parabolic near the goal, conic far away):

```
if dist < D_ATT_THRESHOLD:
    F_att = K_att * dist * d_hat       # parabolic zone: force ∝ distance
else:
    F_att = K_att * D_ATT_THRESHOLD * d_hat   # conic zone: constant force
```

This means:
- **Far from the goal** (dist ≥ `D_ATT_THRESHOLD`): the attractive force is constant, always pulling the robot with full strength.
- **Near the goal** (dist < `D_ATT_THRESHOLD`): the force decreases linearly with distance, giving a smooth, low-overshoot approach.
- The two zones meet continuously at `dist = D_ATT_THRESHOLD`, avoiding any discontinuity.

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

### Local Minimum Escape

A well-known limitation of APF is that the robot can get trapped in **local minima** — configurations where attractive and repulsive forces cancel out, leaving the robot stationary despite not having reached the goal. This happens most often when the robot is caught between two obstacles or cornered by an obstacle directly on the path to the goal.

This simulation includes an automatic escape mechanism that detects stalls and applies a temporary perturbation to break out.

**Detection.** At each step the robot's current speed is measured. If it stays below `STUCK_SPEED_THR` for `STUCK_PATIENCE` consecutive steps, the robot is declared stuck.

**Escape phase.** Once triggered, the escape runs for `ESCAPE_DURATION` steps and applies three simultaneous changes:

1. **Reduces repulsion** — the repulsive force is scaled down to `ESCAPE_REP_SCALE` (default 25%) of its normal magnitude. The robot temporarily accepts a riskier configuration and moves closer to obstacles than it normally would.
2. **Lateral push** — a perturbation is added perpendicular to the goal direction. The sign (+/−) is chosen to point away from the region of strongest repulsion, so the robot sidesteps rather than ramming into the obstacle.
3. **Random noise** — a small random component is added to the lateral direction to break symmetry and avoid the robot oscillating back into the same minimum.

![demo](apf_result_escape.gif)

```
to_goal    = (goal - pos) / ||goal - pos||
perp       = [-to_goal_y, to_goal_x]           # 90° rotation
sign       = +1 if dot(perp, F_rep_dir) ≥ 0 else -1
escape_dir = normalise(sign * perp + noise)

F_total = F_att + ESCAPE_REP_SCALE * F_rep + ESCAPE_PERTURB * escape_dir
```

After the escape phase ends the robot returns to normal APF behaviour. If it gets stuck again the process repeats.

**Visualisation.** During an active escape the robot's trail turns **yellow** and a ⚡ `ESCAPE ACTIVE` badge appears at the top of the panel.



### Emergency Mode

A separate, higher-priority mechanism handles the case where the attractive and repulsive forces are in direct conflict and the robot risks a **collision** — for example when an obstacle moves into the robot from behind, or when the robot is squeezed between the goal direction and an obstacle.

**Trigger.** At every step the simulation measures the distance from the robot to the closest point on the surface of every obstacle. If that distance falls below `D_EMERGENCY`, the emergency mode activates immediately, overriding both normal APF and the escape mechanism.

**Behaviour.** While in emergency mode the robot **completely ignores the goal** and instead runs a real-time **multi-step lookahead** to find the safest escape direction:

1. Sample `N = 24` candidate directions uniformly around 360°.
2. For each direction, simulate `n_steps = 6` time steps forward, advancing obstacle positions to anticipate their movement.
3. Measure the minimum distance to all obstacles **and** world borders along the simulated trajectory (worst-case safety score).
4. Move at `MAX_SPEED` in the direction with the highest worst-case score.

```
for each of N directions:
    score = min distance to obstacles+borders over next n_steps
flee in argmax(score) at MAX_SPEED
```

Because the robot always moves at maximum speed and obstacle speed is capped at `MAX_SPEED × 0.7`, the robot is **guaranteed to gain distance** from any obstacle in at least one direction — making collision avoidance provably effective.

**Hysteresis exit.** To avoid rapid on/off oscillation the robot does not exit emergency mode the moment it crosses `D_EMERGENCY` again. It must reach a larger clearance `D_EMERGENCY_CLEAR` from *all* obstacles before returning to normal or escape state.

```
activate  when:  d_min  ≤  D_EMERGENCY        (1.5)
deactivate when: d_min  ≥  D_EMERGENCY_CLEAR  (2.2)
```

**Obstacle speed limit.** For emergency avoidance to be geometrically guaranteed, obstacle speed is capped at `MAX_SPEED × 0.7 = 1.4`. A faster obstacle could outrun the robot even at full speed, making collision unavoidable regardless of strategy.

**Visualisation.** The trail turns **red** during emergency and a `EMERGENCY — goal ignored` badge appears at the top of the panel.



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
git clone https://github.com/marcopibbes/Robot-Motion-planning-w-Artificial-Potential-Fields.git
cd Robot-Motion-planning-w-Artificial-Potential-Fields

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
Seed = 5  |  4 obstacles
  O1: pos=(9.3, 6.1)   2.8x1.9  vel=(1.23, -0.87)
  O2: pos=(13.5, 14.2) 1.5x2.4  vel=(-0.55, 1.10)
  O3: pos=(6.8, 11.7)  2.1x2.1  STATIONARY
  O4: pos=(11.2, 9.5)  2.3x1.4  vel=(0.45, -1.20)
  [standard] 442 steps  |  escape: 0  emergency: 60  |  final pos=[17.89 17.73]
  [velocity] 398 steps  |  escape: 0  emergency: 45  |  final pos=[17.91 17.75]
```

### Legend (in the animation)

| Symbol | Meaning |
|--------|---------|
| ● green | Start position |
| ✦ gold | Goal position |
| ■ red label | Stationary obstacle |
| ▶ white label | Moving obstacle |
| cyan / orange trail | Robot path (Standard / Velocity-Aware) |
| **yellow trail + ⚡** | Local minimum escape active |
| **red trail + 🚨** | Emergency mode active — goal ignored |

---

## Parameters

All tunable parameters are at the top of `apf_simulation.py`:

### APF core

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
| `D_ATT_THRESHOLD` | 5.0 | Distance threshold between parabolic and conic attractive potential |

### Escape mechanism

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STUCK_SPEED_THR` | 0.08 | Speed below which the robot is considered stuck |
| `STUCK_PATIENCE` | 40 | Consecutive slow steps before escape triggers |
| `ESCAPE_DURATION` | 60 | Steps the escape phase lasts |
| `ESCAPE_REP_SCALE` | 0.25 | Repulsive force multiplier during escape (0 = ignore obstacles, 1 = normal) |
| `ESCAPE_PERTURB` | 1.8 | Magnitude of the lateral perturbation force |


### Emergency mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D_EMERGENCY` | 0.8 | Distance threshold that triggers emergency mode |
| `D_EMERGENCY_CLEAR` | 1.5 | Distance from all obstacles required to exit emergency (hysteresis) |
| `MAX_SPEED` (obs) | `MAX_SPEED × 0.7` | Obstacle speed cap — guarantees robot can always outrun any obstacle |

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
├── apf_simulation.py         # Main simulation script
├── README.md                 # This file
└── apf_result.gif            # Example GIF of a run
└── apf_result_escape.gif     # Example GIF of a run with escape from local minima
└── apf_result_emergency.gif  # Example GIF of a run with emergency mode

```

---

## References

- Khatib, O. (1986). *Real-time obstacle avoidance for manipulators and mobile robots.* The International Journal of Robotics Research, 5(1), 90–98.
- Latombe, J.-C. (1991). *Robot Motion Planning.* Springer.