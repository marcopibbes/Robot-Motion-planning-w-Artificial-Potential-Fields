
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Parametri globali 
WORLD_SIZE  = 20.0
DT          = 0.05
K_ATT       = 10
K_REP       = 80.0
D_INFLUENCE = 3.5
D_GOAL      = 0.3
MAX_SPEED   = 2.0
MAX_STEPS   = 2000

START = np.array([1.0, 1.0])
GOAL  = np.array([18.0, 18.0])

# Potenziale attrattivo
D_ATT_THRESHOLD = 5.0   

# Escape da minimi locali
STUCK_SPEED_THR  = 0.08
STUCK_PATIENCE   = 40
ESCAPE_DURATION  = 60
ESCAPE_REP_SCALE = 0.25
ESCAPE_PERTURB   = 1.8
LOOP_WINDOW      = 50    
LOOP_DIST_THR    = 1   # Se lo spostamento massimo nella finestra è <1m, è in loop


# Emergency (collisione imminente)
D_EMERGENCY         = 0.8 # distanza di pericolo per attivare emergency
D_EMERGENCY_CLEAR   = 2.0   # distanza minima da tutti gli ostacoli per uscire

# Ostacolo 
class MovingObstacle:
    def __init__(self, cx, cy, w, h, vx, vy, color='steelblue'):
        self.cx0, self.cy0 = cx, cy
        self.cx, self.cy   = cx, cy
        self.w, self.h     = w, h
        self.vx0, self.vy0 = vx, vy
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

    def dist_to(self, px, py):
        cp = self.closest_point(px, py)
        return np.linalg.norm(np.array([px, py]) - cp)

    def rect_patch(self, alpha=0.75):
        return patches.Rectangle(
            (self.cx - self.w/2, self.cy - self.h/2),
            self.w, self.h,
            linewidth=1.5, edgecolor='white',
            facecolor=self.color, alpha=alpha
        )

#Generazione ostacoli 
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
            speed = rng.uniform(0.5, MAX_SPEED * 0.7)  # max 70% robot speed: garantisce fuga in emergency
            angle = rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
        obstacles.append(MovingObstacle(cx, cy, w, h, vx, vy, PALETTE[i % len(PALETTE)]))

    return obstacles

# APF 
def attractive_force(pos, goal):
   
    diff = goal - pos
    dist = np.linalg.norm(diff)
    
    # Previene divisione per zero se il robot è esattamente sul goal
    if dist < 1e-6:
        return np.zeros(2)
        
    d_hat = diff / dist  # Versore direzione verso il goal
    
    if dist < D_ATT_THRESHOLD:
        # Zona Quadratica di Huber
        return K_ATT * dist * d_hat
    else:
        # Zona Lineare di Huber
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

def min_obstacle_dist(pos, obstacles):
   
    return min(obs.dist_to(*pos) for obs in obstacles) if obstacles else np.inf

def emergency_flee_direction(pos, obstacles, N=24, n_steps=6):
  
    best_dir   = None
    best_score = -np.inf

    for i in range(N):
        angle = 2 * np.pi * i / N
        d_hat = np.array([np.cos(angle), np.sin(angle)])

        obs_state = [(o.cx, o.cy, o.vx, o.vy) for o in obstacles]
        sim_pos   = pos.copy()
        worst     = np.inf

        for _ in range(n_steps):
            sim_pos = np.clip(sim_pos + d_hat * MAX_SPEED * DT, 0, WORLD_SIZE)
            new_state = []
            for j, (cx, cy, vx, vy) in enumerate(obs_state):
                obs = obstacles[j]
                cx2 = cx + vx * DT;  cy2 = cy + vy * DT
                if cx2 - obs.w/2 < 0 or cx2 + obs.w/2 > WORLD_SIZE:
                    vx = -vx;  cx2 = cx + vx * DT
                if cy2 - obs.h/2 < 0 or cy2 + obs.h/2 > WORLD_SIZE:
                    vy = -vy;  cy2 = cy + vy * DT
                new_state.append((cx2, cy2, vx, vy))
            obs_state = new_state

            d_obs = np.inf
            for j, obs in enumerate(obstacles):
                cx, cy = obs_state[j][0], obs_state[j][1]
                cpx = np.clip(sim_pos[0], cx - obs.w/2, cx + obs.w/2)
                cpy = np.clip(sim_pos[1], cy - obs.h/2, cy + obs.h/2)
                d_obs = min(d_obs, np.linalg.norm(sim_pos - np.array([cpx, cpy])))

            x, y  = sim_pos
            d_bor = min(x, WORLD_SIZE - x, y, WORLD_SIZE - y)
            worst = min(worst, d_obs, d_bor)

        if worst > best_score:
            best_score = worst
            best_dir   = d_hat

    if best_dir is None:
        best_dir = np.array([1.0, 0.0])
    return best_dir * MAX_SPEED

# Simulazione
# Stati del robot
STATE_NORMAL    = 0
STATE_ESCAPE    = 1   # minimo locale
STATE_EMERGENCY = 2   # collisione imminente (priorita' massima)

def simulate(obstacles, mode='standard', rng_escape=None):
    """
    Restituisce:
      path        - array (N,2) posizioni
      obs_history - lista di liste di (cx,cy) per ogni step
      escape_log  - set di step in cui era attivo ESCAPE (minimo locale)
      emerg_log   - set di step in cui era attiva EMERGENCY
    """
    if rng_escape is None:
        rng_escape = np.random.default_rng()

    for obs in obstacles:
        obs.reset()

    pos  = START.copy().astype(float)
    vel  = np.zeros(2)
    path = [pos.copy()]
    obs_history = [[(o.cx, o.cy) for o in obstacles]]

    # stato macchina 
    state          = STATE_NORMAL
    stuck_counter  = 0
    escape_counter = 0
    escape_dir     = np.zeros(2)
    escape_log     = []
    emerg_log      = []
    last_escape_step = -LOOP_WINDOW # NUOVA VARIABILE

    for step in range(MAX_STEPS):
    
        f_att = attractive_force(pos, GOAL)
        if mode == 'standard':
            f_rep = repulsive_force_standard(pos, obstacles)
        else:
            f_rep = repulsive_force_velocity(pos, vel, obstacles)

        # distanza minima dagli ostacoli
        d_min = min_obstacle_dist(pos, obstacles)

        # macchina a stati

        # EMERGENCY ha priorita' assoluta su tutto il resto
        if d_min <= D_EMERGENCY:
            state = STATE_EMERGENCY

        if state == STATE_EMERGENCY:
            emerg_log.append(step)

            # Il robot si muove SEMPRE a MAX_SPEED in quella direzione.
            f_total = emergency_flee_direction(pos, obstacles)
            # uscita dall'emergency solo quando si e' abbastanza lontani 
            if d_min >= D_EMERGENCY_CLEAR:
                state = STATE_NORMAL
                stuck_counter = 0

        elif state == STATE_ESCAPE:
            escape_log.append(step)
            f_rep   = f_rep * ESCAPE_REP_SCALE
            f_total = f_att + f_rep + escape_dir * ESCAPE_PERTURB
            escape_counter -= 1
            stuck_counter   = 0
            if escape_counter <= 0:
                state = STATE_NORMAL

        else:  # STATE_NORMAL
            f_total = f_att + f_rep
            spd_now = np.linalg.norm(vel)
            
            # 1. Rilevamento stuck per bassa velocità
            if spd_now < STUCK_SPEED_THR:
                stuck_counter += 1
            else:
                stuck_counter = 0

            # 2. Rilevamento stuck per loop 
            is_looping = False
        
            if (step - last_escape_step) >= LOOP_WINDOW and len(path) >= LOOP_WINDOW:
                recent_pos = np.array(path[-LOOP_WINDOW:])
                # Calcola la diagonale del bounding box delle posizioni recenti
                max_spread = np.linalg.norm(np.max(recent_pos, axis=0) - np.min(recent_pos, axis=0))
                if max_spread < LOOP_DIST_THR:
                    is_looping = True

            # Trigger dell'escape
            if stuck_counter >= STUCK_PATIENCE or is_looping:
                # transizione a ESCAPE
                last_escape_step = step 
                
                to_goal   = GOAL - pos
                to_goal  /= np.linalg.norm(to_goal) + 1e-9
                perp      = np.array([-to_goal[1], to_goal[0]])
                f_rep_dir = f_rep / (np.linalg.norm(f_rep) + 1e-9)
                sign      = 1.0 if np.dot(perp, f_rep_dir) >= 0 else -1.0
                noise     = rng_escape.uniform(-0.3, 0.3, size=2)
                escape_dir  = sign * perp + noise
                escape_dir /= np.linalg.norm(escape_dir) + 1e-9
                escape_counter = ESCAPE_DURATION
                stuck_counter  = 0
                state = STATE_ESCAPE

        # aggiorna posizione
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

    return np.array(path), obs_history, escape_log, emerg_log

# Animazioni
def run_animation(seed=None, gif_path=None):
    modes  = ['standard', 'velocity']
    titles = ['APF Standard', 'APF + Velocita\' Relativa']
    colors = ['#00d4ff', '#ff6b35']

    obstacles = make_obstacles(seed=seed)
    n_obs     = len(obstacles)

    print(f"Seed = {seed if seed is not None else 'random'}  |  {n_obs} ostacoli")
    for i, o in enumerate(obstacles):
        stato = "FERMO" if (o.vx0==0 and o.vy0==0) else f"vel=({o.vx0:.2f},{o.vy0:.2f})"
        print(f"  O{i+1}: pos=({o.cx0:.1f},{o.cy0:.1f})  {o.w:.1f}x{o.h:.1f}  {stato}")

    rng_escape = np.random.default_rng(None if seed is None else seed + 1000)

    paths, obs_histories, escape_logs, emerg_logs = [], [], [], []
    for m in modes:
        p, oh, esc, emg = simulate(obstacles, mode=m, rng_escape=rng_escape)
        paths.append(p)
        obs_histories.append(oh)
        escape_logs.append(set(esc))
        emerg_logs.append(set(emg))
        print(f"  [{m}] {len(p)} step  |  escape: {len(esc)}  emergency: {len(emg)}  |  fine={p[-1].round(2)}")

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

        obs_patches, obs_labels = [], []
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

        # trail normale
        trail,        = ax.plot([], [], '-', color=colors[i], lw=1.5, alpha=0.7, zorder=5)
        robot,        = ax.plot([], [], 'o', color=colors[i], ms=9,   zorder=10)
        # trail escape (giallo)
        escape_trail, = ax.plot([], [], '-', color='#ffdd00', lw=2.5, alpha=0.85, zorder=6)
        escape_dot,   = ax.plot([], [], 's', color='#ffdd00', ms=7,   zorder=11)
        # trail emergency (rosso)
        emerg_trail,  = ax.plot([], [], '-', color='#ff2222', lw=3.0, alpha=0.9,  zorder=7)
        emerg_dot,    = ax.plot([], [], 'D', color='#ff2222', ms=9,   zorder=12)

        info = ax.text(0.02, 0.02, '', transform=ax.transAxes, color='white',
                       fontsize=8, va='bottom',
                       bbox=dict(boxstyle='round', facecolor='#111', alpha=0.75))

        # badge escape
        esc_badge = ax.text(0.5, 0.93, '\u26a1 ESCAPE ACTIVE',
                            transform=ax.transAxes, color='#ffdd00',
                            fontsize=10, fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round', facecolor='#222', alpha=0.85),
                            visible=False, zorder=20)
        # badge emergency 
        emg_badge = ax.text(0.5, 0.93, '\u26A0 EMERGENCY — goal ignored',
                            transform=ax.transAxes, color='#ff4444',
                            fontsize=10, fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round', facecolor='#300', alpha=0.9),
                            visible=False, zorder=21)

        ax.text(0.98, 0.02,
                '\u25a0 stationary  \u25b6 moving\n'
                '\u26a1 escape (yellow)  \u26A0 emergency (red)',
                transform=ax.transAxes, color='#aaa', fontsize=7,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='#111', alpha=0.6))

        panels.append((trail, robot,
                        escape_trail, escape_dot, esc_badge,
                        emerg_trail,  emerg_dot,  emg_badge,
                        obs_patches, obs_labels, info,
                        paths[i], obs_histories[i],
                        escape_logs[i], emerg_logs[i]))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    max_frames = max(len(p) for p in paths)

    def update(frame):
        artists = []
        for (trail, robot,
             escape_trail, escape_dot, esc_badge,
             emerg_trail,  emerg_dot,  emg_badge,
             obs_patches, obs_labels, info,
             path, obs_hist,
             escape_set, emerg_set) in panels:

            f    = min(frame, len(path) - 1)
            oh_f = min(frame, len(obs_hist) - 1)
            px, py = path[f]

           
            norm_x, norm_y = [], []
            esc_x,  esc_y  = [], []
            emg_x,  emg_y  = [], []
            for k in range(f + 1):
                in_emg = k in emerg_set
                in_esc = k in escape_set
                norm_x.append(path[k,0] if not in_esc and not in_emg else np.nan)
                norm_y.append(path[k,1] if not in_esc and not in_emg else np.nan)
                esc_x.append(path[k,0]  if in_esc and not in_emg      else np.nan)
                esc_y.append(path[k,1]  if in_esc and not in_emg      else np.nan)
                emg_x.append(path[k,0]  if in_emg                     else np.nan)
                emg_y.append(path[k,1]  if in_emg                     else np.nan)

            trail.set_data(norm_x, norm_y)
            escape_trail.set_data(esc_x, esc_y)
            emerg_trail.set_data(emg_x,  emg_y)

            # robot markers
            in_emg_now = f in emerg_set
            in_esc_now = f in escape_set and not in_emg_now

            robot.set_visible(not in_esc_now and not in_emg_now)
            robot.set_data([px], [py])

            escape_dot.set_visible(in_esc_now)
            escape_dot.set_data([px], [py])

            emerg_dot.set_visible(in_emg_now)
            emerg_dot.set_data([px], [py])

            # badge: emergency ha priorita' su escape
            emg_badge.set_visible(in_emg_now)
            esc_badge.set_visible(in_esc_now and not in_emg_now)

            # ostacoli
            centers = obs_hist[oh_f]
            for j, (patch, lbl, obs) in enumerate(zip(obs_patches, obs_labels, obstacles)):
                cx, cy = centers[j]
                patch.set_xy((cx - obs.w/2, cy - obs.h/2))
                lbl.set_position((cx, cy))

            d  = np.linalg.norm(path[f] - GOAL)
            st = "ARRIVATO!" if d < D_GOAL * 3 else f"dist goal: {d:.2f}"
            info.set_text(f"step {f}/{len(path)-1}  {st}")

            artists += [trail, robot,
                        escape_trail, escape_dot, esc_badge,
                        emerg_trail,  emerg_dot,  emg_badge,
                        info] + obs_patches + obs_labels
        return artists

    ani = FuncAnimation(fig, update, frames=max_frames,
                        interval=25, blit=True, repeat=True)

    if gif_path:
        print(f"Salvataggio GIF: {gif_path}  ({max_frames} frame)...")
        ani.save(gif_path, writer='pillow', fps=30,
                 savefig_kwargs={'facecolor': '#0d1117'})
        print(f"GIF salvata in: {gif_path}")
        plt.close(fig)
    else:
        plt.show()

    return ani

# main
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