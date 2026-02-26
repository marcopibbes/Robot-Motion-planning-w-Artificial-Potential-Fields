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
# Soglia entro cui il potenziale attrattivo e' parabolico (proporzionale a d).
# Oltre questa soglia diventa conico (forza costante = K_ATT * D_ATT_THRESHOLD).
# Cio' evita forze enormi lontano dal goal (instabilita') e forze nulle vicino
# (convergenza lenta), ed e' lo schema standard nella letteratura APF.
D_ATT_THRESHOLD = 5.0   # [unita' mondo] soglia parabola/cono

def attractive_force(pos, goal):
    """
    Potenziale attrattivo ibrido (posizione-dipendente):
      dist < D_ATT_THRESHOLD  ->  forza lineare in dist  (parabolico, morbido)
      dist >= D_ATT_THRESHOLD ->  forza costante K_ATT   (conico, spinge sempre)
    Quando l'agente e' lontano dal goal riceve una forza piena;
    quando si avvicina la forza decresce proporzionalmente, evitando overshooting.
    """
    diff = goal - pos
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np.zeros(2)
    d_hat = diff / dist
    if dist < D_ATT_THRESHOLD:
        # zona parabolica: forza proporzionale alla distanza
        return K_ATT * dist * d_hat
    else:
        # zona conica: forza costante (uguale al valore al confine per continuita')
        return K_ATT * D_ATT_THRESHOLD * d_hat

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

# ─── Escape da minimi locali ─────────────────────────────────────────────────
# Se il robot e' fermo (o quasi) per STUCK_PATIENCE step consecutivi,
# si attiva una "escape perturbation" che dura ESCAPE_DURATION step.
# Durante l'escape, la forza repulsiva e' scalata di ESCAPE_REP_SCALE
# (< 1 => meno paura degli ostacoli) e viene aggiunta una spinta laterale
# casuale per uscire dal minimo locale, mantenendo pero' la direzione
# generale verso il goal.

STUCK_SPEED_THR  = 0.08   # velocita' media sotto cui il robot e' considerato bloccato
STUCK_PATIENCE   = 40     # step consecutivi lenti prima di attivare escape
ESCAPE_DURATION  = 60     # step durante cui l'escape e' attivo
ESCAPE_REP_SCALE = 0.25   # fattore di riduzione della forza repulsiva durante escape
ESCAPE_PERTURB   = 1.8    # ampiezza della spinta laterale casuale

# ─── Simulazione ─────────────────────────────────────────────────────────────
def simulate(obstacles, mode='standard', rng_escape=None):
    """
    Simula usando una lista di ostacoli GIA' CREATA.
    Gli ostacoli vengono resettati alla posizione iniziale prima della simulazione.
    Restituisce path, obs_history e escape_log (lista di step in cui era attivo escape).
    """
    if rng_escape is None:
        rng_escape = np.random.default_rng()

    for obs in obstacles:
        obs.reset()

    pos  = START.copy().astype(float)
    vel  = np.zeros(2)
    path = [pos.copy()]
    obs_history = [[(o.cx, o.cy) for o in obstacles]]

    stuck_counter  = 0          # step consecutivi con bassa velocita'
    escape_counter = 0          # step rimanenti di escape attivo
    escape_dir     = np.zeros(2)# direzione laterale di perturbazione fissa per tutto l'escape
    escape_log     = []         # step in cui escape e' stato attivo (per visualizzazione)

    for step in range(MAX_STEPS):
        # ── forze base ──────────────────────────────────────────────────────
        f_att = attractive_force(pos, GOAL)
        if mode == 'standard':
            f_rep = repulsive_force_standard(pos, obstacles)
        else:
            f_rep = repulsive_force_velocity(pos, vel, obstacles)

        # ── logica escape ────────────────────────────────────────────────────
        spd_now = np.linalg.norm(vel)
        if escape_counter > 0:
            # escape attivo: riduci repulsione e aggiungi spinta laterale verso goal
            f_rep    = f_rep * ESCAPE_REP_SCALE
            f_total  = f_att + f_rep + escape_dir * ESCAPE_PERTURB
            escape_counter -= 1
            escape_log.append(step)
            stuck_counter = 0
        else:
            f_total = f_att + f_rep
            # controlla se il robot e' bloccato
            if spd_now < STUCK_SPEED_THR:
                stuck_counter += 1
            else:
                stuck_counter = 0

            if stuck_counter >= STUCK_PATIENCE:
                # attiva escape: calcola una direzione laterale rispetto al goal
                to_goal  = GOAL - pos
                to_goal  = to_goal / (np.linalg.norm(to_goal) + 1e-9)
                perp     = np.array([-to_goal[1], to_goal[0]])  # perpendicolare
                # scegli il verso (+/-) che allontana di piu' dagli ostacoli vicini
                f_rep_dir = f_rep / (np.linalg.norm(f_rep) + 1e-9)
                sign      = 1.0 if np.dot(perp, f_rep_dir) >= 0 else -1.0
                # aggiungi piccolo rumore per evitare oscillazioni simmetriche
                noise        = rng_escape.uniform(-0.3, 0.3, size=2)
                escape_dir   = sign * perp + noise
                escape_dir  /= np.linalg.norm(escape_dir) + 1e-9
                escape_counter = ESCAPE_DURATION
                stuck_counter  = 0

        # ── aggiorna posizione ───────────────────────────────────────────────
        spd = np.linalg.norm(f_total)
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

    return np.array(path), obs_history, escape_log

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

    # Seme per l'escape (riproducibile se seed e' dato)
    rng_escape = np.random.default_rng(None if seed is None else seed + 1000)

    # Simula entrambe le modalita' sugli STESSI ostacoli
    paths, obs_histories, escape_logs = [], [], []
    for m in modes:
        p, oh, esc = simulate(obstacles, mode=m, rng_escape=rng_escape)
        paths.append(p)
        obs_histories.append(oh)
        escape_logs.append(set(esc))
        n_esc = len(esc)
        print(f"  [{m}] {len(p)} step  |  escape attivati: {n_esc}  |  fine pos={p[-1].round(2)}")

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
        ax.text(0.98, 0.02,
                '\u25a0 stationary  \u25b6 moving\n\u26a1 escape active (yellow)',
                transform=ax.transAxes, color='#aaa', fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='#111', alpha=0.6))

        # linea escape separata (colorata diversamente)
        escape_trail, = ax.plot([], [], '-', color='#ffdd00', lw=2.5,
                                alpha=0.85, zorder=6)
        escape_dot,   = ax.plot([], [], 's', color='#ffdd00', ms=7, zorder=11)

        # badge "ESCAPE" (visibile solo quando attivo)
        esc_badge = ax.text(0.5, 0.93, '⚡ ESCAPE ACTIVE',
                            transform=ax.transAxes, color='#ffdd00',
                            fontsize=10, fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round', facecolor='#333', alpha=0.85),
                            visible=False, zorder=20)

        panels.append((trail, robot, obs_patches, obs_labels, info,
                        escape_trail, escape_dot, esc_badge,
                        paths[i], obs_histories[i], escape_logs[i]))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    max_frames = max(len(p) for p in paths)

    def update(frame):
        artists = []
        for (trail, robot, obs_patches, obs_labels, info,
             escape_trail, escape_dot, esc_badge,
             path, obs_hist, escape_set) in panels:

            f    = min(frame, len(path) - 1)
            oh_f = min(frame, len(obs_hist) - 1)

            px, py = path[f]

            # ── trail normale (tratti NON in escape) ──────────────────────
            norm_x, norm_y = [], []
            esc_x,  esc_y  = [], []
            for k in range(f + 1):
                if k in escape_set:
                    esc_x.append(path[k, 0]);  esc_y.append(path[k, 1])
                    # interrompi trail normale
                    norm_x.append(np.nan);      norm_y.append(np.nan)
                else:
                    norm_x.append(path[k, 0]); norm_y.append(path[k, 1])
                    esc_x.append(np.nan);       esc_y.append(np.nan)

            trail.set_data(norm_x, norm_y)
            escape_trail.set_data(esc_x, esc_y)

            # ── robot marker: giallo se in escape, normale altrimenti ─────
            in_escape_now = f in escape_set
            robot.set_data([px], [py])
            if in_escape_now:
                escape_dot.set_data([px], [py])
                escape_dot.set_visible(True)
                robot.set_visible(False)
            else:
                escape_dot.set_visible(False)
                robot.set_visible(True)

            esc_badge.set_visible(in_escape_now)

            # ── ostacoli ──────────────────────────────────────────────────
            centers = obs_hist[oh_f]
            for j, (patch, lbl, obs) in enumerate(zip(obs_patches, obs_labels, obstacles)):
                cx, cy = centers[j]
                patch.set_xy((cx - obs.w/2, cy - obs.h/2))
                lbl.set_position((cx, cy))

            d  = np.linalg.norm(path[f] - GOAL)
            st = "ARRIVATO!" if d < D_GOAL * 3 else f"dist goal: {d:.2f}"
            info.set_text(f"step {f}/{len(path)-1}  {st}")
            artists += [trail, robot, escape_trail, escape_dot,
                        esc_badge, info] + obs_patches + obs_labels
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