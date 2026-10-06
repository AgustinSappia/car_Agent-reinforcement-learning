"""
Modo fútbol: los autos aprenden a llevar la pelota al arco rival.

- La cancha es una "pista" del editor con dos arcos (o la cancha clásica incluida).
- Se juegan muchos partidos a la vez, cada uno invisible para los demás. Se muestra uno.
- Cada cerebro maneja a todos los jugadores de su equipo (cada uno con sus propias entradas).
- El rival puede ser un bot programado, otro cerebro de la población (aprenden jugando
  entre ellos) o nadie (práctica de goles).

Puntaje de un equipo en un partido:
  goles a favor x1000 - goles en contra x600 + toques de pelota + cuánto acercó la pelota
  al arco rival + qué tan cerca estuvo de la pelota.
"""

import math

import numpy as np
import pygame

from ai.world import World, surface_mask

TEAM_NAMES = ('azul', 'rojo')
TEAM_COLORS = ((60, 140, 255), (235, 70, 70))
TEAM_LIGHT = ((150, 195, 255), (255, 160, 160))

BALL_R = 10
CAR_R = 13
KICK_RANGE = 36
KICK_POWER = 8.5
KICK_COOLDOWN = 25
BALL_MAX_SPEED = 14.0
BALL_FRICTION = 0.985
WALL_BOUNCE = 0.75

GRASS = (46, 125, 60)
GRASS_DARK = (40, 112, 54)
WALL_DRAW = (24, 26, 34)

OPPONENTS = [
    ('bot_facil', "Bot fácil"),
    ('bot_normal', "Bot normal"),
    ('bot_dificil', "Bot difícil"),
    ('self', "Entre ellos"),
    ('none', "Sin rival"),
]
BOT_SPEED = {'bot_facil': 0.55, 'bot_normal': 0.8, 'bot_dificil': 1.0}

FIELD_W, FIELD_H = 1570, 1080


# ---------------------------------------------------------------------- #
# Cancha
# ---------------------------------------------------------------------- #
def default_field_surface():
    """Cancha clásica: rectángulo redondeado con un arco a cada lado"""
    surf = pygame.Surface((FIELD_W, FIELD_H))
    surf.fill((0, 0, 0))
    white = (255, 255, 255)
    pygame.draw.rect(surf, white, pygame.Rect(110, 120, 1350, 840), border_radius=120)
    pygame.draw.rect(surf, white, pygame.Rect(40, 440, 80, 200))
    pygame.draw.rect(surf, white, pygame.Rect(1450, 440, 80, 200))
    goals = {'azul': (40, 440, 112, 640), 'rojo': (1458, 440, 1530, 640)}
    return surf, goals, (785, 540)


class Field:
    def __init__(self, track_data=None):
        if track_data is None:
            surf, goals, ball = default_field_surface()
            self.name, self.file = "Cancha clásica", ''
        else:
            surf = track_data['track_layer']
            goals = track_data.get('goals') or {}
            ball = track_data.get('ball_spawn')
            self.name = track_data.get('name', 'cancha')
            self.file = track_data.get('file', self.name)
        self.road_mask = surface_mask(surf)
        self.goal_rects = []
        for team in TEAM_NAMES:
            x1, y1, x2, y2 = goals[team]
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            self.goal_rects.append((float(x1), float(y1), float(x2), float(y2)))
            self.road_mask[int(x1):int(x2) + 1, int(y1):int(y2) + 1] = True
        self.goal_centers = np.array([((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in self.goal_rects],
                                     dtype=np.float32)
        if ball is None:
            ball = self.goal_centers.mean(axis=0)
        self.ball_spawn = (float(ball[0]), float(ball[1]))
        self.background = self._draw_background()

    @property
    def size(self):
        return self.road_mask.shape

    def on_road(self, x, y):
        W, H = self.road_mask.shape
        return 0 <= x < W and 0 <= y < H and self.road_mask[int(x), int(y)]

    def team_spawns(self, team, count):
        """Posiciones de saque: entre la pelota y el arco propio, mirando a la pelota"""
        bx, by = self.ball_spawn
        gx, gy = self.goal_centers[team]
        dx, dy = bx - gx, by - gy
        length = math.hypot(dx, dy) or 1
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        spots = []
        for k in range(count):
            off = (k - (count - 1) / 2) * 110
            frac = 0.5 if count == 1 else 0.42
            x, y = gx + dx * (1 - frac) + px * off, gy + dy * (1 - frac) + py * off
            # Si cayó fuera de la cancha, acercarlo a la pelota
            for _ in range(40):
                if self.on_road(x, y):
                    break
                x, y = x + (bx - x) * 0.08, y + (by - y) * 0.08
            spots.append((float(x), float(y), float(math.atan2(by - y, bx - x))))
        return spots

    def _draw_background(self):
        W, H = self.road_mask.shape
        cols = np.where((np.arange(W) // 80) % 2 == 0, 1, 0)[:, None]
        grass = np.where(cols[..., None] == 1, np.array(GRASS), np.array(GRASS_DARK))
        img = np.where(self.road_mask[..., None], grass, np.array(WALL_DRAW)).astype(np.uint8)
        bg = pygame.surfarray.make_surface(img)
        line = (225, 235, 225)
        # Línea central y círculo, perpendiculares a la línea entre arcos
        (ax, ay), (bx, by) = self.goal_centers.tolist()
        cx, cy = self.ball_spawn
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy) or 1
        px, py = -dy / length, dx / length
        pygame.draw.line(bg, line, (cx - px * 2000, cy - py * 2000), (cx + px * 2000, cy + py * 2000), 3)
        pygame.draw.circle(bg, line, (int(cx), int(cy)), 110, 3)
        pygame.draw.circle(bg, line, (int(cx), int(cy)), 6)
        # Volver a pintar las paredes encima de las líneas
        pixels = pygame.surfarray.pixels3d(bg)
        pixels[~self.road_mask] = WALL_DRAW
        del pixels
        # Arcos con red
        for team, r in enumerate(self.goal_rects):
            rect = pygame.Rect(int(r[0]), int(r[1]), int(r[2] - r[0]), int(r[3] - r[1]))
            net = pygame.Surface(rect.size, pygame.SRCALPHA)
            net.fill(TEAM_COLORS[team] + (90,))
            for k in range(0, max(rect.w, rect.h) * 2, 12):
                pygame.draw.line(net, (255, 255, 255, 120), (k, 0), (k - rect.h, rect.h), 1)
                pygame.draw.line(net, (255, 255, 255, 120), (k - rect.h, 0), (k, rect.h), 1)
            bg.blit(net, rect)
            pygame.draw.rect(bg, TEAM_COLORS[team], rect, 4)
        return bg


# ---------------------------------------------------------------------- #
# Simulación de muchos partidos a la vez
# ---------------------------------------------------------------------- #
class FootballSim:
    """
    M partidos en paralelo. Autos ordenados por partido: el auto c es del partido c // PPM,
    del equipo (c % PPM) // team_size. PPM = jugadores por partido.
    """

    def __init__(self, field, cfg, n_matches, team_size, two_teams=True, rng=None):
        self.field = field
        self.cfg = cfg
        self.M = n_matches
        self.N = team_size
        self.teams = 2 if two_teams else 1
        self.PPM = team_size * self.teams
        self.C = self.M * self.PPM
        self.rng = rng or np.random.default_rng()
        self.world = World(field.road_mask, cfg, kind='futbol')
        self.world.reset(self.C, 0, 0, 0)
        self.actions = np.array(cfg.actions('futbol'), dtype=np.float32)

        c = np.arange(self.C)
        self.match_of = c // self.PPM
        self.team_of = (c % self.PPM) // self.N
        self.spawns = [field.team_spawns(t, self.N) for t in range(2)]
        self.goal_c = field.goal_centers
        self.goal_r = np.array(field.goal_rects, dtype=np.float32)
        self.reset_all()

    # -------------------------------------------------------------- #
    def reset_all(self):
        M = self.M
        self.bx = np.zeros(M, dtype=np.float32)
        self.by = np.zeros(M, dtype=np.float32)
        self.bvx = np.zeros(M, dtype=np.float32)
        self.bvy = np.zeros(M, dtype=np.float32)
        self.score = np.zeros((M, 2), dtype=np.int32)
        self.touches = np.zeros(self.C, dtype=np.int32)
        self.kicks = np.zeros(self.C, dtype=np.int32)
        self.goals_by = np.zeros(self.C, dtype=np.int32)
        self.cooldown = np.zeros(self.C, dtype=np.int32)
        self.touching = np.zeros(self.C, dtype=bool)
        self.last_touch = np.full(M, -1, dtype=np.int32)
        self.advance = np.zeros((M, 2), dtype=np.float32)     # mejor avance de la pelota hacia cada arco rival
        self.near = np.zeros((M, 2), dtype=np.float32)        # suma de cercanía a la pelota
        self.steps = 0
        self.last_goal = np.full(M, -1000, dtype=np.int32)
        self.kickoff(np.arange(M))

    def kickoff(self, matches):
        if len(matches) == 0:
            return
        self.bx[matches], self.by[matches] = self.field.ball_spawn
        self.bvx[matches] = 0
        self.bvy[matches] = 0
        # Distancia inicial de la pelota a cada arco, para medir avances
        for t in range(self.teams):
            for k in range(self.N):
                cars = matches * self.PPM + t * self.N + k
                x, y, a = self.spawns[t][k]
                n = len(cars)
                self.world.place(cars, x + self.rng.uniform(-8, 8, n), y + self.rng.uniform(-8, 8, n),
                                 a + self.rng.uniform(-0.2, 0.2, n))
        self.cooldown[np.isin(self.match_of, matches)] = 0
        self.d0 = np.hypot(self.field.ball_spawn[0] - self.goal_c[:, 0], self.field.ball_spawn[1] - self.goal_c[:, 1])

    def opp_goal(self, cars):
        return self.goal_c[1 - self.team_of[cars]]

    def own_goal(self, cars):
        return self.goal_c[self.team_of[cars]]

    # -------------------------------------------------------------- #
    def _rel(self, cars, tx, ty, scale):
        w = self.world
        dx, dy = tx - w.x[cars], ty - w.y[cars]
        rel = np.arctan2(dy, dx) - w.angle[cars]
        return [np.sin(rel), np.cos(rel), np.minimum(np.hypot(dx, dy) / scale, 1.5)]

    def nearest(self, cars, same_team):
        """Posición del compañero o rival más cercano de cada auto (o None si no hay)"""
        w = self.world
        m = self.match_of[cars]
        base = m * self.PPM
        best_d = np.full(len(cars), 1e9, dtype=np.float32)
        bx = np.zeros(len(cars), dtype=np.float32)
        by = np.zeros(len(cars), dtype=np.float32)
        for j in range(self.PPM):
            other = base + j
            ok = (other != cars) & ((self.team_of[other] == self.team_of[cars]) == same_team)
            d = np.hypot(w.x[other] - w.x[cars], w.y[other] - w.y[cars])
            better = ok & (d < best_d)
            best_d[better] = d[better]
            bx[better] = w.x[other][better]
            by[better] = w.y[other][better]
        found = best_d < 1e8
        return bx, by, found

    def observe(self, cars):
        """Entradas de la red de cada auto: todo relativo al frente del auto"""
        w, cfg = self.world, self.cfg
        m = self.match_of[cars]
        parts = [w.read_sensors(cars)]
        if cfg.use_speed:
            parts.append(w.speed[cars] / cfg.max_speed)
        parts += self._rel(cars, self.bx[m], self.by[m], 1000)
        ca, sa = np.cos(-w.angle[cars]), np.sin(-w.angle[cars])
        parts += [(self.bvx[m] * ca - self.bvy[m] * sa) / 10, (self.bvx[m] * sa + self.bvy[m] * ca) / 10]
        og = self.opp_goal(cars)
        parts += self._rel(cars, og[:, 0], og[:, 1], 1500)
        mg = self.own_goal(cars)
        parts += self._rel(cars, mg[:, 0], mg[:, 1], 1500)[:2]
        for same in (True, False):
            nx, ny, found = self.nearest(cars, same)
            s, c, d = self._rel(cars, nx, ny, 1000)
            parts += [np.where(found, s, 0), np.where(found, c, 0), np.where(found, d, 1.5)]
        parts.append((self.cooldown[cars] == 0).astype(np.float32))
        cols = [p if p.ndim == 2 else p[:, None] for p in parts]
        return np.concatenate(cols, axis=1).astype(np.float32)

    def bot_controls(self, cars):
        """Bot programado: se pone detrás de la pelota mirando al arco rival y la empuja"""
        w = self.world
        m = self.match_of[cars]
        bx, by = self.bx[m], self.by[m]
        og = self.opp_goal(cars)
        gx, gy = og[:, 0] - bx, og[:, 1] - by
        gl = np.hypot(gx, gy) + 1e-6
        ux, uy = gx / gl, gy / gl
        # Punto detrás de la pelota
        tx, ty = bx - ux * (BALL_R + CAR_R + 8), by - uy * (BALL_R + CAR_R + 8)
        cx, cy = w.x[cars], w.y[cars]
        to_ball_x, to_ball_y = bx - cx, by - cy
        dist_ball = np.hypot(to_ball_x, to_ball_y) + 1e-6
        behind = (to_ball_x * ux + to_ball_y * uy) / dist_ball > 0.6
        tx = np.where(behind, bx, tx)
        ty = np.where(behind, by, ty)
        want = np.arctan2(ty - cy, tx - cx)
        diff = (want - w.angle[cars] + math.pi) % (2 * math.pi) - math.pi
        steer = np.where(np.abs(diff) > 0.1, np.sign(diff), 0).astype(np.float32)
        throttle = np.where(np.abs(diff) < 1.0, 1.0, 0.25).astype(np.float32)
        brake = np.zeros(len(cars), dtype=np.float32)
        facing_goal = np.cos(w.angle[cars]) * ux + np.sin(w.angle[cars]) * uy > 0.85
        kick = (behind & facing_goal & (dist_ball < KICK_RANGE + BALL_R)).astype(np.float32)
        return steer, throttle, brake, kick

    def action_controls(self, action_idx):
        a = self.actions[action_idx]
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    # -------------------------------------------------------------- #
    def step(self, steer, throttle, brake, kick):
        """Un paso de todos los partidos. Controles: arrays de largo C. Devuelve los partidos con gol."""
        w = self.world
        cars = np.arange(self.C)
        px, py = w.x.copy(), w.y.copy()
        crashed = w.step_raw(cars, steer, throttle, brake)
        # Contra la pared: vuelve a donde estaba y frena
        w.x[crashed], w.y[crashed] = px[crashed], py[crashed]
        w.speed[crashed] = 0
        self._car_collisions()
        self._ball_contacts(kick)
        goals = self._move_ball()
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        self._accumulate()
        self.steps += 1
        return goals

    def _car_collisions(self):
        w = self.world
        before_x, before_y = w.x.copy(), w.y.copy()
        X = w.x.reshape(self.M, self.PPM)
        Y = w.y.reshape(self.M, self.PPM)
        for i in range(self.PPM):
            for j in range(i + 1, self.PPM):
                dx, dy = X[:, j] - X[:, i], Y[:, j] - Y[:, i]
                d = np.hypot(dx, dy) + 1e-6
                over = np.maximum(2 * CAR_R - d, 0) / 2
                hit = over > 0
                if hit.any():
                    nx, ny = dx / d * over, dy / d * over
                    X[hit, i] -= nx[hit]
                    Y[hit, i] -= ny[hit]
                    X[hit, j] += nx[hit]
                    Y[hit, j] += ny[hit]
        # Si el empujón lo sacó de la cancha, vuelve a donde estaba
        off = ~w.on_road(w.x, w.y)
        w.x[off], w.y[off] = before_x[off], before_y[off]

    def _ball_contacts(self, kick):
        w = self.world
        m = self.match_of
        dx, dy = self.bx[m] - w.x, self.by[m] - w.y
        d = np.hypot(dx, dy) + 1e-6
        nx, ny = dx / d, dy / d
        ca, sa = np.cos(w.angle), np.sin(w.angle)

        # Empujón: la pelota sale despedida en la dirección del contacto
        touch = d < CAR_R + BALL_R
        new_touch = touch & ~self.touching
        self.touches += new_touch
        self.touching = touch
        if touch.any():
            c = np.flatnonzero(touch)
            mm = m[c]
            push = w.speed[c] * 1.15 + 0.6
            vn = self.bvx[mm] * nx[c] + self.bvy[mm] * ny[c]
            add = np.maximum(push - vn, 0)
            self.bvx[mm] += nx[c] * add
            self.bvy[mm] += ny[c] * add
            # Separar la pelota del auto (si no queda contra la pared)
            sx = w.x[c] + nx[c] * (CAR_R + BALL_R + 0.5)
            sy = w.y[c] + ny[c] * (CAR_R + BALL_R + 0.5)
            free = w.on_road(sx, sy) & w.on_road(sx + nx[c] * BALL_R, sy + ny[c] * BALL_R)
            self.bx[mm[free]] = sx[free]
            self.by[mm[free]] = sy[free]
            # Pelota atrapada contra la pared: rebota hacia el auto que la empuja
            stuck = mm[~free]
            self.bvx[stuck] *= -0.5
            self.bvy[stuck] *= -0.5
            self.last_touch[mm] = c

        # Patada: la pelota tiene que estar cerca y adelante del auto
        front = (nx * ca + ny * sa) > 0.55
        can = (kick > 0) & (self.cooldown == 0) & front & (d < KICK_RANGE + BALL_R)
        self.cooldown[(kick > 0) & (self.cooldown == 0)] = KICK_COOLDOWN // 2
        if can.any():
            c = np.flatnonzero(can)
            mm = m[c]
            self.bvx[mm] += ca[c] * KICK_POWER
            self.bvy[mm] += sa[c] * KICK_POWER
            self.cooldown[c] = KICK_COOLDOWN
            self.kicks[c] += 1
            self.last_touch[mm] = c
        speed = np.hypot(self.bvx, self.bvy)
        too_fast = speed > BALL_MAX_SPEED
        self.bvx[too_fast] *= BALL_MAX_SPEED / speed[too_fast]
        self.bvy[too_fast] *= BALL_MAX_SPEED / speed[too_fast]

    def _move_ball(self):
        w = self.world
        for _ in range(2):  # dos subpasos para que no atraviese paredes
            vx, vy = self.bvx / 2, self.bvy / 2
            nx = self.bx + vx
            hit_x = ~w.on_road(nx + np.sign(vx) * BALL_R, self.by)
            self.bvx[hit_x] *= -WALL_BOUNCE
            self.bx = np.where(hit_x, self.bx, nx)
            ny = self.by + vy
            hit_y = ~w.on_road(self.bx, ny + np.sign(vy) * BALL_R)
            self.bvy[hit_y] *= -WALL_BOUNCE
            self.by = np.where(hit_y, self.by, ny)
        self.bvx *= BALL_FRICTION
        self.bvy *= BALL_FRICTION

        # Gol: la pelota entra en un arco. Arco 0 (azul) -> gol del equipo rojo (1)
        scored = np.full(self.M, -1, dtype=np.int32)
        for g in range(2):
            x1, y1, x2, y2 = self.goal_r[g]
            inside = (self.bx > x1) & (self.bx < x2) & (self.by > y1) & (self.by < y2)
            scored[inside] = 1 - g
        goal_m = np.flatnonzero(scored >= 0)
        if len(goal_m):
            team = scored[goal_m]
            if self.teams == 1:
                team = np.where(team == 1, 0, -1)  # sin rival: en contra propia no suma a nadie
            ok = team >= 0
            self.score[goal_m[ok], team[ok]] += 1
            if self.teams == 1:
                self.score[goal_m[~ok], 1] += 1
            scorer = self.last_touch[goal_m]
            good = (scorer >= 0)
            self.goals_by[scorer[good]] += (self.team_of[scorer[good]] == scored[goal_m][good])
            self.last_goal[goal_m] = self.steps
            self.kickoff(goal_m)
        return goal_m

    def _accumulate(self):
        """Datos para el puntaje: avance de la pelota hacia cada arco y cercanía a la pelota"""
        for t in range(self.teams):
            target = self.goal_c[1 - t]
            d = np.hypot(self.bx - target[0], self.by - target[1])
            self.advance[:, t] = np.maximum(self.advance[:, t], self.d0[1 - t] - d)
            cars = (np.arange(self.M)[:, None] * self.PPM + t * self.N + np.arange(self.N)[None, :])
            dist = np.hypot(self.world.x[cars] - self.bx[:, None], self.world.y[cars] - self.by[:, None]).min(axis=1)
            self.near[:, t] += 1 - np.minimum(dist / 800, 1)

    def team_fitness(self):
        """(M, equipos) puntaje de cada equipo en su partido"""
        fit = np.zeros((self.M, self.teams), dtype=np.float32)
        for t in range(self.teams):
            cars = (np.arange(self.M)[:, None] * self.PPM + t * self.N + np.arange(self.N)[None, :])
            touches = np.minimum(self.touches[cars].sum(axis=1), 30)
            against = self.score[:, 1 - t] if self.teams == 2 else self.score[:, 1]
            fit[:, t] = (self.score[:, t] * 1000 - against * 600 + touches * 15
                         + np.maximum(self.advance[:, t], 0) + 200 * self.near[:, t] / max(1, self.steps))
        return fit


# ---------------------------------------------------------------------- #
# Escenario para el entrenador
# ---------------------------------------------------------------------- #
class FootballScenario:
    key = 'futbol'
    title = 'Fútbol'
    goal_text = "Meter la pelota en el arco rival"

    def __init__(self, field, team_size=1, opponent='bot_normal', match_steps=1500):
        self.field = field
        self.team_size = team_size
        self.opponent = opponent
        self.match_steps = match_steps
        self.changed = True
        sx, sy, sa = field.team_spawns(0, 1)[0]
        self.spawn = (sx, sy, sa)

    # Lo que usa el taller del agente para la vista previa
    @property
    def road_mask(self):
        return self.field.road_mask

    @property
    def background(self):
        return self.field.background

    @property
    def size(self):
        return self.field.size

    @property
    def name(self):
        return self.field.name

    def opponent_label(self):
        return dict(OPPONENTS)[self.opponent]

    def status(self):
        return [f"Rival: {self.opponent_label()}  ·  {self.team_size} por equipo", f"Cancha: {self.field.name}"]

    def draw_overlay(self, surf, to_screen, scale, font):
        pass

    def map_info(self):
        return {'map': self.field.name, 'map_file': self.field.file, 'team_size': self.team_size,
                'opponent': self.opponent, 'match_steps': self.match_steps}


# ---------------------------------------------------------------------- #
# Dibujo
# ---------------------------------------------------------------------- #
def draw_match(surf, sim, match, to_screen, scale, labels=None, highlight=None):
    """Dibuja los autos y la pelota de un partido"""
    w = sim.world
    L, Wd = 18 * scale, 10 * scale
    for k in range(sim.PPM):
        c = match * sim.PPM + k
        team = sim.team_of[c]
        cx, cy = to_screen((w.x[c], w.y[c]))
        ang = float(w.angle[c])
        ca, sa = math.cos(ang), math.sin(ang)
        pts = [(cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)
               for dx, dy in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd))]
        pygame.draw.polygon(surf, TEAM_COLORS[team], pts)
        pygame.draw.polygon(surf, TEAM_LIGHT[team], pts, 2)
        nose = (cx + ca * L, cy + sa * L)
        pygame.draw.circle(surf, (255, 255, 255), (int(nose[0]), int(nose[1])), max(2, int(3 * scale)))
        if highlight is not None and c == highlight:
            pygame.draw.circle(surf, (255, 205, 40), (int(cx), int(cy)), int(max(10, 26 * scale)), 2)
        if labels and c in labels:
            font = pygame.font.Font(None, 22)
            t = font.render(labels[c], True, (255, 255, 255))
            surf.blit(t, t.get_rect(midbottom=(cx, cy - 20 * scale)))
    bx, by = to_screen((sim.bx[match], sim.by[match]))
    r = max(4, int(BALL_R * scale))
    pygame.draw.circle(surf, (250, 250, 250), (int(bx), int(by)), r)
    pygame.draw.circle(surf, (20, 20, 20), (int(bx), int(by)), r, 2)


def draw_score(surf, font, center, score, names=("AZUL", "ROJO")):
    a, b = int(score[0]), int(score[1])
    txt = font.render(f"{names[0]}  {a}  -  {b}  {names[1]}", True, (255, 255, 255))
    r = txt.get_rect(center=center)
    pygame.draw.rect(surf, (20, 22, 30), r.inflate(30, 14), border_radius=10)
    left = r.inflate(30, 14)
    pygame.draw.rect(surf, TEAM_COLORS[0], (left.x, left.y, 10, left.h), border_radius=4)
    pygame.draw.rect(surf, TEAM_COLORS[1], (left.right - 10, left.y, 10, left.h), border_radius=4)
    surf.blit(txt, r)
