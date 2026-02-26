"""
APF (Artificial Potential Fields) Simulation with Moving Obstacles
==================================================================
Caso 1: APF Standard
Caso 2: APF con fattore moltiplicativo basato sulla velocità relativa
         F_rep *= (1 + |v_rel|) -- quando ostacolo fermo: |v_rel|=0 => fattore=1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# ─── Parametri globali ───────────────────────────────────────────────────────
WORLD_SIZE   = 20.0
DT           = 0.05          # passo temporale
K_ATT        = 1.5           # guadagno forza attrattiva
K_REP        = 80.0          # guadagno forza repulsiva
D_INFLUENCE  = 3.5           # distanza di influenza ostacoli
D_GOAL       = 0.3           # distanza di arrivo al goal
MAX_SPEED    = 2.0           # velocità massima robot
MAX_STEPS    = 2000

START = np.array([1.0, 1.0])
GOAL  = np.array([18.0, 18.0])

# ─── Definizione ostacoli ────────────────────────────────────────────────────
class MovingObstacle:
    """Ostacolo rettangolare in movimento."""
    def __init__(self, cx, cy, w, h, vx, vy, color='steelblue'):
        self.cx, self.cy = cx, cy
        self.w, self.h   = w, h
        self.vx, self.vy = vx, vy   # velocità ostacolo
        self.color = color

    def step(self):
        self.cx += self.vx * DT
        self.cy += self.vy * DT
        # rimbalzo ai bordi
        if self.cx - self.w/2 < 0 or self.cx + self.w/2 > WORLD_SIZE:
            self.vx = -self.vx
        if self.cy - self.h/2 < 0 or self.cy + self.h/2 > WORLD_SIZE:
            self.vy = -self.vy

    def polygon(self):
        cx, cy, w, h = self.cx, self.cy, self.w, self.h
        return Polygon([
            (cx-w/2, cy-h/2), (cx+w/2, cy-h/2),
            (cx+w/2, cy+h/2), (cx-w/2, cy+h/2)
        ])

    def closest_point(self, px, py):
        """Punto più vicino sulla superficie del rettangolo."""
        cx = np.clip(px, self.cx - self.w/2, self.cx + self.w/2)
        cy = np.clip(py, self.cy - self.h/2, self.cy + self.h/2)
        return np.array([cx, cy])

    def rect_patch(self, alpha=0.7):
        return patches.Rectangle(
            (self.cx - self.w/2, self.cy - self.h/2),
            self.w, self.h,
            linewidth=1.5, edgecolor='white',
            facecolor=self.color, alpha=alpha
        )


def make_obstacles():
    return [
        MovingObstacle(6,  10, 2.0, 3.0,  1.5,  0.0, '#4a90d9'),  # si muove orizzontalmente
        MovingObstacle(12, 5,  2.5, 2.5,  0.0,  1.2, '#e07b39'),  # si muove verticalmente
        MovingObstacle(10, 14, 3.0, 1.5, -1.0,  0.8, '#7bc67e'),  # diagonale
        MovingObstacle(16, 8,  2.0, 2.0,  0.0,  0.0, '#c97bd1'),  # FERMO (v=0 => fattore=1)
    ]

# ─── APF Core ────────────────────────────────────────────────────────────────

def attractive_force(pos, goal):
    diff = goal - pos
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np.zeros(2)
    return K_ATT * diff / dist   # normalizzata

def repulsive_force_standard(pos, obstacles):
    """APF standard: solo distanza."""
    total = np.zeros(2)
    for obs in obstacles:
        cp = obs.closest_point(*pos)
        diff = pos - cp
        dist = np.linalg.norm(diff)
        if dist < 1e-6 or dist > D_INFLUENCE:
            continue
        mag = K_REP * (1.0/dist - 1.0/D_INFLUENCE) / (dist**2)
        total += mag * diff / dist
    return total

def repulsive_force_velocity(pos, robot_vel, obstacles):
    """APF variante: moltiplicato per (1 + |v_rel|).
       Quando ostacolo fermo: v_rel = v_robot => fattore = 1 + |v_robot|.
       PERÒ: vogliamo fattore=1 quando ostacolo fermo rispetto al robot.
       Interpretiamo: v_rel = v_ostacolo - v_robot
       fattore = 1 + max(0, -v_rel · d̂)  (componente verso il robot)
       oppure più semplice: fattore = 1 + |v_obs| (velocità dell'ostacolo).
       Quando l'ostacolo è fermo => |v_obs|=0 => fattore=1. ✓
    """
    total = np.zeros(2)
    for obs in obstacles:
        cp = obs.closest_point(*pos)
        diff = pos - cp
        dist = np.linalg.norm(diff)
        if dist < 1e-6 or dist > D_INFLUENCE:
            continue
        v_obs = np.array([obs.vx, obs.vy])
        v_rel = v_obs - robot_vel                  # velocità relativa
        # fattore: 1 + componente di v_rel verso il robot (se avvicinante)
        d_hat = diff / dist
        approach = np.dot(v_rel, -d_hat)           # positivo se ostacolo si avvicina
        factor = 1.0 + max(0.0, approach)          # ≥1; =1 se ostacolo fermo/allontana
        mag = K_REP * (1.0/dist - 1.0/D_INFLUENCE) / (dist**2)
        total += factor * mag * diff / dist
    return total


def simulate(mode='standard'):
    """Simula il percorso del robot. mode: 'standard' | 'velocity'"""
    obstacles = make_obstacles()
    pos  = START.copy().astype(float)
    vel  = np.zeros(2)
    path = [pos.copy()]
    obs_history = [[np.array([o.cx, o.cy]) for o in obstacles]]

    for _ in range(MAX_STEPS):
        f_att = attractive_force(pos, GOAL)
        if mode == 'standard':
            f_rep = repulsive_force_standard(pos, obstacles)
        else:
            f_rep = repulsive_force_velocity(pos, vel, obstacles)

        f_total = f_att + f_rep
        # limita velocità
        speed = np.linalg.norm(f_total)
        if speed > MAX_SPEED:
            f_total = f_total / speed * MAX_SPEED

        vel = f_total
        pos = pos + vel * DT

        # clamp al mondo
        pos = np.clip(pos, 0, WORLD_SIZE)
        path.append(pos.copy())

        for obs in obstacles:
            obs.step()
        obs_history.append([np.array([o.cx, o.cy]) for o in obstacles])

        if np.linalg.norm(pos - GOAL) < D_GOAL:
            break

    return np.array(path), obs_history, obstacles

# ─── Visualizzazione ─────────────────────────────────────────────────────────

def run_animation():
    fig = plt.figure(figsize=(14, 7), facecolor='#0d1117')
    fig.suptitle('APF — Artificial Potential Fields con Ostacoli Mobili',
                 color='white', fontsize=14, fontweight='bold', y=0.97)

    axes = []
    titles = ['APF Standard', 'APF + Velocità Relativa']
    modes  = ['standard', 'velocity']
    colors = ['#00d4ff', '#ff6b35']

    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, facecolor='#0d1117')
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal')
        ax.set_title(titles[i], color=colors[i], fontsize=12, pad=8)
        ax.tick_params(colors='#555')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        # griglia sottile
        ax.grid(True, color='#1e2a3a', linewidth=0.5, alpha=0.6)
        axes.append(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Simula entrambi
    paths, obs_histories, final_obs_list = [], [], []
    for m in modes:
        p, oh, fo = simulate(m)
        paths.append(p)
        obs_histories.append(oh)
        final_obs_list.append(fo)

    # Ostacoli originali per shape/size (ricreiamo per avere posizioni iniziali)
    base_obs = make_obstacles()

    # Elementi grafici per ogni pannello
    panels = []
    for i, (ax, path, obs_hist, fobs) in enumerate(zip(axes, paths, obs_histories, final_obs_list)):
        color = colors[i]
        # start / goal markers
        ax.plot(*START, 'o', color='#7fff7f', ms=10, zorder=10)
        ax.text(START[0]+0.3, START[1]+0.3, 'START', color='#7fff7f', fontsize=8)
        ax.plot(*GOAL,  '*', color='#ffd700', ms=14, zorder=10)
        ax.text(GOAL[0]+0.3, GOAL[1]+0.3, 'GOAL', color='#ffd700', fontsize=8)

        # traccia percorso (statica di sfondo, aggiornata)
        trail,  = ax.plot([], [], '-', color=color, lw=1.2, alpha=0.6, zorder=5)
        robot,  = ax.plot([], [], 'o', color=color, ms=8, zorder=10)

        # Ostacoli - patches (num_obs)
        n_obs = len(base_obs)
        obs_patches = []
        for obs in base_obs:
            p = obs.rect_patch(alpha=0.75)
            ax.add_patch(p)
            obs_patches.append(p)

        # testo info
        info_txt = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                           color='white', fontsize=8, va='bottom',
                           bbox=dict(boxstyle='round', facecolor='#111', alpha=0.7))

        panels.append((trail, robot, obs_patches, info_txt, path, obs_hist))

    # Numero di frame = lunghezza massima percorso
    max_frames = max(len(p) for p in paths)

    def update(frame):
        artists = []
        for i, (trail, robot, obs_patches, info_txt, path, obs_hist) in enumerate(panels):
            f = min(frame, len(path)-1)
            oh_f = min(frame, len(obs_hist)-1)

            # posizione robot
            px, py = path[f]
            trail.set_data(path[:f+1, 0], path[:f+1, 1])
            robot.set_data([px], [py])

            # ostacoli
            obs_centers = obs_hist[oh_f]
            for j, (patch, base) in enumerate(zip(obs_patches, base_obs)):
                cx, cy = obs_centers[j]
                patch.set_xy((cx - base.w/2, cy - base.h/2))

            dist_goal = np.linalg.norm(path[f] - GOAL)
            status = "ARRIVATO!" if dist_goal < D_GOAL*2 else f"dist goal: {dist_goal:.2f}"
            info_txt.set_text(f"step {f}/{len(path)-1} | {status}")

            artists += [trail, robot, info_txt] + obs_patches
        return artists

    ani = FuncAnimation(fig, update, frames=max_frames,
                        interval=30, blit=True, repeat=True)

    plt.show()
    return ani


if __name__ == '__main__':
    ani = run_animation()