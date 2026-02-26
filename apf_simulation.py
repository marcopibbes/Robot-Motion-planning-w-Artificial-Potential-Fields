"""
APF (Artificial Potential Fields) Simulation with Moving Obstacles
==================================================================
Caso 1: APF Standard
Caso 2: APF + Velocita' Relativa

Esegui con seed fisso:  python apf_simulation.py --seed 42
Esegui random:          python apf_simulation.py
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# ─── Parametri globali ───────────────────────────────────────────────────────
WORLD_SIZE  = 20.0
DT          = 0.05
K_ATT       = 1.5
K_REP       = 80.0
D_INFLUENCE = 3.5
D_GOAL      = 0.3
MAX_SPEED   = 2.0
MAX_STEPS   = 2000

START = np.array([1.0, 1.0])
GOAL  = np.array([18.0, 18.0])

# ─── Ostacolo ────────────────────────────────────────────────────────────────
class MovingObstacle:
    def __init__(self, cx, cy, w, h, vx, vy, color='steelblue'):
        self.cx0, self.cy0 = cx, cy   # posizione iniziale (per reset)
        self.cx, self.cy   = cx, cy
        self.w, self.h     = w, h
        self.vx0, self.vy0 = vx, vy   # velocita' iniziale (per reset)
        self.vx, self.vy   = vx, vy
        self.color         = color

    def reset(self):
        self.cx, self.cy = self.cx0, self.cy0
        self.vx, self.vy = self.vx0, self.vy0

    def step(self):
        self.cx += self.vx * DT
        self.cy += self.vy * DT
        if self.cx - self.w/2 < 0 or self.cx + self.w/2 > WORLD_SIZE:
            self.vx = -self.vx
        if self.cy - self.h/2 < 0 or self.cy + self.h/2 > WORLD_SIZE:
            self.vy = -self.vy

    def closest_point(self, px, py):
        cx = np.clip(px, self.cx - self.w/2, self.cx + self.w/2)
        cy = np.clip(py, self.cy - self.h/2, self.cy + self.h/2)
        return np.array([cx, cy])

    def rect_patch(self, alpha=0.75):
        return patches.Rectangle(
            (self.cx - self.w/2, self.cy - self.h/2),
            self.w, self.h,
            linewidth=1.5, edgecolor='white',
            facecolor=self.color, alpha=alpha
        )

# ─── Generazione ostacoli randomizzata ───────────────────────────────────────
def make_obstacles(seed=None):
    rng = np.random.default_rng(seed)
    PALETTE = ['#4a90d9','#e07b39','#7bc67e','#c97bd1',
               '#f0c040','#e05c7a','#50d0c0','#ff9966']
    MARGIN = 3.5
    n_obs = int(rng.integers(3, 8))
    obstacles = []

    for i in range(n_obs):
        w = rng.uniform(1.2, 3.2)
        h = rng.uniform(1.2, 3.2)
        for _ in range(300):
            cx = rng.uniform(w/2 + 0.5, WORLD_SIZE - w/2 - 0.5)
            cy = rng.uniform(h/2 + 0.5, WORLD_SIZE - h/2 - 0.5)
            ok_start = np.linalg.norm([cx-START[0], cy-START[1]]) > MARGIN + max(w, h)
            ok_goal  = np.linalg.norm([cx-GOAL[0],  cy-GOAL[1]])  > MARGIN + max(w, h)
            if ok_start and ok_goal:
                break
        if i == n_obs - 1 or rng.random() < 0.25:
            vx, vy = 0.0, 0.0
        else:
            speed = rng.uniform(0.5, 2.2)
            angle = rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
        obstacles.append(MovingObstacle(cx, cy, w, h, vx, vy, PALETTE[i % len(PALETTE)]))

    return obstacles

# ─── APF Core ────────────────────────────────────────────────────────────────
def attractive_force(pos, goal):
    diff = goal - pos
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np.zeros(2)
    return K_ATT * diff / dist

def repulsive_force_standard(pos, obstacles):
    total = np.zeros(2)
    for obs in obstacles:
        cp   = obs.closest_point(*pos)
        diff = pos - cp
        dist = np.linalg.norm(diff)
        if dist < 1e-6 or dist > D_INFLUENCE:
            continue
        mag   = K_REP * (1.0/dist - 1.0/D_INFLUENCE) / (dist**2)
        total += mag * diff / dist
    return total

def repulsive_force_velocity(pos, robot_vel, obstacles):
    total = np.zeros(2)
    for obs in obstacles:
        cp   = obs.closest_point(*pos)
        diff = pos - cp
        dist = np.linalg.norm(diff)
        if dist < 1e-6 or dist > D_INFLUENCE:
            continue
        d_hat    = diff / dist
        v_obs    = np.array([obs.vx, obs.vy])
        v_rel    = v_obs - robot_vel
        approach = np.dot(v_rel, -d_hat)
        factor   = 1.0 + max(0.0, approach)
        mag      = K_REP * (1.0/dist - 1.0/D_INFLUENCE) / (dist**2)
        total   += factor * mag * diff / dist
    return total

# ─── Simulazione ─────────────────────────────────────────────────────────────
def simulate(obstacles, mode='standard'):
    """
    Simula usando una lista di ostacoli GIA' CREATA.
    Gli ostacoli vengono resettati alla posizione iniziale prima della simulazione.
    Restituisce path e obs_history con lo stesso numero di ostacoli garantito.
    """
    # reset ostacoli alla posizione/velocita' iniziale
    for obs in obstacles:
        obs.reset()

    pos  = START.copy().astype(float)
    vel  = np.zeros(2)
    path = [pos.copy()]
    # salva (cx, cy, vx, vy) per poter aggiornare correttamente le patch
    obs_history = [[(o.cx, o.cy) for o in obstacles]]

    for _ in range(MAX_STEPS):
        f_att = attractive_force(pos, GOAL)
        if mode == 'standard':
            f_rep = repulsive_force_standard(pos, obstacles)
        else:
            f_rep = repulsive_force_velocity(pos, vel, obstacles)

        f_total = f_att + f_rep
        spd     = np.linalg.norm(f_total)
        if spd > MAX_SPEED:
            f_total = f_total / spd * MAX_SPEED

        vel  = f_total
        pos  = np.clip(pos + vel * DT, 0, WORLD_SIZE)
        path.append(pos.copy())

        for obs in obstacles:
            obs.step()
        obs_history.append([(o.cx, o.cy) for o in obstacles])

        if np.linalg.norm(pos - GOAL) < D_GOAL:
            break

    return np.array(path), obs_history

# ─── Animazione ──────────────────────────────────────────────────────────────
def run_animation(seed=None, gif_path=None):
    modes  = ['standard', 'velocity']
    titles = ['APF Standard', 'APF + Velocita\' Relativa']
    colors = ['#00d4ff', '#ff6b35']

    # Crea UN solo set di ostacoli condiviso, poi resetta per ogni simulazione
    obstacles = make_obstacles(seed=seed)
    n_obs     = len(obstacles)

    print(f"Seed = {seed if seed is not None else 'random'}  |  {n_obs} ostacoli")
    for i, o in enumerate(obstacles):
        stato = "FERMO" if (o.vx0==0 and o.vy0==0) else f"vel=({o.vx0:.2f},{o.vy0:.2f})"
        print(f"  O{i+1}: pos=({o.cx0:.1f},{o.cy0:.1f})  {o.w:.1f}x{o.h:.1f}  {stato}")

    # Simula entrambe le modalita' sugli STESSI ostacoli
    paths, obs_histories = [], []
    for m in modes:
        p, oh = simulate(obstacles, mode=m)
        paths.append(p)
        obs_histories.append(oh)
        print(f"  [{m}] {len(p)} step  |  fine pos={p[-1].round(2)}")

    # ── Figura ──
    fig = plt.figure(figsize=(15, 7.5), facecolor='#0d1117')
    seed_label = f"seed={seed}" if seed is not None else "seed=random"
    fig.suptitle(f'APF con Ostacoli Randomizzati  [{seed_label}  |  {n_obs} ostacoli]',
                 color='white', fontsize=13, fontweight='bold', y=0.97)

    panels = []
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, facecolor='#0d1117')
        ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal')
        ax.set_title(titles[i], color=colors[i], fontsize=12, pad=8)
        ax.tick_params(colors='#555')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.grid(True, color='#1e2a3a', linewidth=0.5, alpha=0.5)

        ax.plot(*START, 'o', color='#7fff7f', ms=10, zorder=10)
        ax.text(START[0]+0.3, START[1]+0.3, 'START', color='#7fff7f', fontsize=8, fontweight='bold')
        ax.plot(*GOAL, 'P', color='#ffd700', ms=12, zorder=10)
        ax.text(GOAL[0]+0.3, GOAL[1]+0.3, 'GOAL', color='#ffd700', fontsize=8, fontweight='bold')

        obs_patches = []
        obs_labels  = []
        for obs in obstacles:
            patch = obs.rect_patch()
            ax.add_patch(patch)
            is_fermo = (obs.vx0 == 0 and obs.vy0 == 0)
            lbl = ax.text(obs.cx0, obs.cy0,
                          '\u25a0' if is_fermo else '\u25b6',
                          color='#ff4444' if is_fermo else 'white',
                          ha='center', va='center', fontsize=9, zorder=6)
            obs_patches.append(patch)
            obs_labels.append(lbl)

        trail, = ax.plot([], [], '-', color=colors[i], lw=1.5, alpha=0.7, zorder=5)
        robot, = ax.plot([], [], 'o', color=colors[i], ms=9, zorder=10)
        info   = ax.text(0.02, 0.02, '', transform=ax.transAxes, color='white',
                         fontsize=8, va='bottom',
                         bbox=dict(boxstyle='round', facecolor='#111', alpha=0.75))
        ax.text(0.98, 0.02, '\u25a0 fermo  \u25b6 mobile',
                transform=ax.transAxes, color='#aaa', fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='#111', alpha=0.6))

        panels.append((trail, robot, obs_patches, obs_labels, info,
                        paths[i], obs_histories[i]))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    max_frames = max(len(p) for p in paths)

    def update(frame):
        artists = []
        for (trail, robot, obs_patches, obs_labels, info, path, obs_hist) in panels:
            f    = min(frame, len(path) - 1)
            oh_f = min(frame, len(obs_hist) - 1)

            px, py = path[f]
            trail.set_data(path[:f+1, 0], path[:f+1, 1])
            robot.set_data([px], [py])

            centers = obs_hist[oh_f]   # lista di (cx,cy) lunga n_obs
            for j, (patch, lbl, obs) in enumerate(zip(obs_patches, obs_labels, obstacles)):
                cx, cy = centers[j]
                patch.set_xy((cx - obs.w/2, cy - obs.h/2))
                lbl.set_position((cx, cy))

            d  = np.linalg.norm(path[f] - GOAL)
            st = "ARRIVATO!" if d < D_GOAL * 3 else f"dist goal: {d:.2f}"
            info.set_text(f"step {f}/{len(path)-1}  {st}")
            artists += [trail, robot, info] + obs_patches + obs_labels
        return artists

    ani = FuncAnimation(fig, update, frames=max_frames,
                        interval=25, blit=True, repeat=True)

    if gif_path:
        print(f"Salvataggio GIF: {gif_path}  ({max_frames} frame, potrebbe richiedere qualche minuto)...")
        ani.save(gif_path, writer='pillow', fps=30,
                 savefig_kwargs={'facecolor': '#0d1117'})
        print(f"GIF salvata in: {gif_path}")
        plt.close(fig)
    else:
        plt.show()

    return ani

# ─── Entry point ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APF con ostacoli randomizzati')
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed per riproducibilita' (es. --seed 42). "
                             "Ometti per scenario casuale ad ogni run.")
    parser.add_argument('--gif', type=str, default=None, metavar='FILE.gif',
                        help="Salva l'animazione come GIF (es. --gif risultato.gif). "
                             "Richiede Pillow: pip install pillow")
    args = parser.parse_args()
    run_animation(seed=args.seed, gif_path=args.gif)