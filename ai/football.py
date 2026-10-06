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
IDLE_SPEED = 0.3       # más lento que esto cuenta como quieto
CROWD_DIST = 70        # dos compañeros más cerca que esto están amontonados
HOLD_SPEED = 0.85      # con la pelota pegada, el auto anda a este porcentaje de su velocidad máxima
GOOD_KICK_COS = 0.85   # una patada es "útil" si la pelota sale a menos de ~30° del arco rival
BALL_CONTROLS = [('pegada', "Pegada"), ('libre', "Libre")]

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
        # Hasta 3 jugadores: una línea. De 4 a 6: defensa (cerca del arco) y ataque
        if count <= 3:
            lines = [(count, 0.5 if count == 1 else 0.42)]
        else:
            front = count // 2
            lines = [(count - front, 0.72), (front, 0.36)]
        spots = []
        for n_line, frac in lines:
            for k in range(n_line):
                off = (k - (n_line - 1) / 2) * 115
                spots.append((gx + dx * (1 - frac) + px * off, gy + dy * (1 - frac) + py * off))
        result = []
        for x, y in spots:
            # Si cayó fuera de la cancha, acercarlo a la pelota
            for _ in range(40):
                if self.on_road(x, y):
                    break
                x, y = x + (bx - x) * 0.08, y + (by - y) * 0.08
            result.append((float(x), float(y), float(math.atan2(by - y, bx - x))))
        return result

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

    def __init__(self, field, cfg, n_matches, team_size, two_teams=True, rng=None, control='libre'):
        self.field = field
        # 'pegada': al tocar la pelota de frente queda pegada al auto hasta que patea, choca o se la roba un rival
        self.sticky = control == 'pegada'
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
        self.role = ((c % self.N) / max(1, self.N - 1)).astype(np.float32)   # 0 = primero, 1 = último
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
        self.good_kicks = np.zeros(self.C, dtype=np.int32)      # patadas que salen hacia el arco rival
        self.owner = np.full(M, -1, dtype=np.int32)             # auto que tiene la pelota pegada (o -1)
        self.grab_cd = np.zeros(self.C, dtype=np.int32)         # pasos hasta poder volver a agarrarla
        self.goals_by = np.zeros(self.C, dtype=np.int32)
        self.cooldown = np.zeros(self.C, dtype=np.int32)
        self.touching = np.zeros(self.C, dtype=bool)
        self.last_touch = np.full(M, -1, dtype=np.int32)
        self.advance = np.zeros((M, 2), dtype=np.float32)     # mejor avance de la pelota hacia cada arco rival
        self.near = np.zeros((M, 2), dtype=np.float32)        # suma de cercanía a la pelota
        self.own_goals = np.zeros((M, 2), dtype=np.int32)     # goles en contra hechos por el propio equipo
        self.crowd = np.zeros((M, 2), dtype=np.float32)       # pasos con compañeros pegados
        self.idle = np.zeros((M, 2), dtype=np.float32)        # pasos quietos (promedio del equipo)
        self.steps = 0
        self.last_goal = np.full(M, -1000, dtype=np.int32)
        self.kickoff(np.arange(M))

    def kickoff(self, matches):
        if len(matches) == 0:
            return
        self.bx[matches], self.by[matches] = self.field.ball_spawn
        self.bvx[matches] = 0
        self.bvy[matches] = 0
        self.owner[matches] = -1
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
        for same, wanted in ((True, cfg.see_mates), (False, cfg.see_rivals)):
            if not wanted:
                continue
            nx, ny, found = self.nearest(cars, same)
            s, c, d = self._rel(cars, nx, ny, 1000)
            parts += [np.where(found, s, 0), np.where(found, c, 0), np.where(found, d, 1.5)]
        if cfg.use_role:
            parts.append(self.role[cars])
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
        # En equipos de 2 o más: solo el atacante más cercano va a la pelota, los otros atacantes
        # acompañan un poco atrás y la mitad del equipo defiende entre la pelota y su arco.
        # Cada uno se corre hacia un costado según su número, para no amontonarse.
        if self.N > 1:
            k = cars % self.N
            lat = (k - (self.N - 1) / 2) * 75
            perp_x, perp_y = -uy, ux
            mg = self.own_goal(cars)
            defender = k >= (self.N + 1) // 2
            far = np.hypot(bx - mg[:, 0], by - mg[:, 1]) > 350
            hold = defender & far
            team_key = m * 2 + self.team_of[cars]
            d = np.where(hold, np.inf, dist_ball)
            nearest = np.full(self.M * 2, np.inf)
            np.minimum.at(nearest, team_key, d)
            support = ~hold & (d > nearest[team_key] + 1e-3)
            tx = np.where(support, bx - ux * 140 + perp_x * lat, tx)
            ty = np.where(support, by - uy * 140 + perp_y * lat, ty)
            tx = np.where(hold, mg[:, 0] + (bx - mg[:, 0]) * 0.35 + perp_x * lat, tx)
            ty = np.where(hold, mg[:, 1] + (by - mg[:, 1]) * 0.35 + perp_y * lat, ty)
        # Con la pelota pegada: va hacia el arco y patea cuando está cerca y mirándolo
        has = (self.owner[m] == cars) if self.sticky else np.zeros(len(cars), dtype=bool)
        tx = np.where(has, og[:, 0], tx)
        ty = np.where(has, og[:, 1], ty)
        want = np.arctan2(ty - cy, tx - cx)
        diff = (want - w.angle[cars] + math.pi) % (2 * math.pi) - math.pi
        steer = np.where(np.abs(diff) > 0.1, np.sign(diff), 0).astype(np.float32)
        throttle = np.where(np.abs(diff) < 1.0, 1.0, 0.25).astype(np.float32)
        throttle[np.hypot(tx - cx, ty - cy) < 24] = 0   # ya llegó a su lugar
        brake = np.zeros(len(cars), dtype=np.float32)
        facing_goal = np.cos(w.angle[cars]) * ux + np.sin(w.angle[cars]) * uy > 0.85
        kick = behind & facing_goal & (dist_ball < KICK_RANGE + BALL_R)
        if self.sticky:
            near_goal = np.hypot(og[:, 0] - cx, og[:, 1] - cy) < 420
            kick = np.where(has, facing_goal & near_goal, kick & False)
        kick = kick.astype(np.float32)
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
        if self.sticky:
            o = self.owner[self.owner >= 0]
            w.speed[o] = np.minimum(w.speed[o], HOLD_SPEED * self.cfg.max_speed * w.speed_mult[o])
        self._car_collisions()
        self._ball_contacts(kick)
        goals = self._move_ball()
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        self.grab_cd = np.maximum(self.grab_cd - 1, 0)
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
        busy = self._hold(kick, d) if self.sticky else np.zeros(self.M, dtype=bool)

        # Empujón: la pelota sale despedida en la dirección del contacto
        touch = (d < CAR_R + BALL_R) & ~busy[m]
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
        can = (kick > 0) & (self.cooldown == 0) & front & (d < KICK_RANGE + BALL_R) & ~busy[m]
        self.cooldown[(kick > 0) & (self.cooldown == 0)] = KICK_COOLDOWN // 2
        if can.any():
            c = np.flatnonzero(can)
            mm = m[c]
            self.bvx[mm] += ca[c] * KICK_POWER
            self.bvy[mm] += sa[c] * KICK_POWER
            self._after_kick(c, mm)
        if self.sticky:
            self._grab(d, front, busy)
        speed = np.hypot(self.bvx, self.bvy)
        too_fast = speed > BALL_MAX_SPEED
        self.bvx[too_fast] *= BALL_MAX_SPEED / speed[too_fast]
        self.bvy[too_fast] *= BALL_MAX_SPEED / speed[too_fast]

    def _after_kick(self, c, mm):
        self.cooldown[c] = KICK_COOLDOWN
        self.kicks[c] += 1
        self.last_touch[mm] = c
        # ¿Sale hacia el arco rival?
        g = self.opp_goal(c)
        gx, gy = g[:, 0] - self.bx[mm], g[:, 1] - self.by[mm]
        v = np.hypot(self.bvx[mm], self.bvy[mm]) * np.hypot(gx, gy) + 1e-6
        good = (self.bvx[mm] * gx + self.bvy[mm] * gy) / v > GOOD_KICK_COS
        self.good_kicks[c[good]] += 1

    def _release(self, matches, cooldown):
        o = self.owner[matches]
        self.grab_cd[o] = cooldown
        self.owner[matches] = -1

    def _hold(self, kick, d):
        """Pelota pegada: robos, patadas y llevarla adelante del auto. Devuelve los partidos que
        ya no hay que tocar en este paso (la pelota la tiene alguien o se acaba de patear)."""
        w, m = self.world, self.match_of
        busy = np.zeros(self.M, dtype=bool)
        if not (self.owner >= 0).any():
            return busy
        # Robo: un rival toca la pelota y se suelta
        owner_team = self.team_of[np.maximum(self.owner, 0)][m]
        steal = (self.owner[m] >= 0) & (self.team_of != owner_team) & (d < CAR_R + BALL_R + 2)
        if steal.any():
            self._release(np.unique(m[steal]), 30)
        held = np.flatnonzero(self.owner >= 0)
        o = self.owner[held]
        ca, sa = np.cos(w.angle[o]), np.sin(w.angle[o])
        fx = w.x[o] + ca * (CAR_R + BALL_R + 1)
        fy = w.y[o] + sa * (CAR_R + BALL_R + 1)
        # Patada: sale disparada hacia adelante
        shoot = (kick[o] > 0) & (self.cooldown[o] == 0)
        if shoot.any():
            mm, c = held[shoot], o[shoot]
            self.bvx[mm] = ca[shoot] * (KICK_POWER + w.speed[c])
            self.bvy[mm] = sa[shoot] * (KICK_POWER + w.speed[c])
            self._after_kick(c, mm)
            self._release(mm, 20)
            busy[mm] = True
        # El resto la lleva adelante; si adelante hay pared, se le escapa
        keep = ~shoot
        free = w.on_road(fx, fy) & w.on_road(fx + ca * BALL_R, fy + sa * BALL_R)
        ok = keep & free
        mm = held[ok]
        self.bx[mm], self.by[mm] = fx[ok], fy[ok]
        self.bvx[mm] = ca[ok] * w.speed[o[ok]]
        self.bvy[mm] = sa[ok] * w.speed[o[ok]]
        busy[mm] = True
        lost = held[keep & ~free]
        if len(lost):
            self.bvx[lost] = 0
            self.bvy[lost] = 0
            self._release(lost, 15)
        return busy

    def _grab(self, d, front, busy):
        """Un auto que toca la pelota de frente se queda con ella (el más cercano de cada partido)"""
        m = self.match_of
        cand = np.flatnonzero((self.grab_cd == 0) & front & (d < CAR_R + BALL_R + 4)
                              & (self.owner[m] < 0) & ~busy[m])
        if not len(cand):
            return
        best = np.full(self.M, np.inf)
        np.minimum.at(best, m[cand], d[cand])
        win = cand[d[cand] <= best[m[cand]]]
        mm, first = np.unique(m[win], return_index=True)
        self.owner[mm] = win[first]
        self.last_touch[mm] = win[first]
        self.touches[win[first]] += 1

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
                team = np.where(team == 0, 0, -1)  # sin rival: el gol en contra propia no suma a nadie
            ok = team >= 0
            self.score[goal_m[ok], team[ok]] += 1
            if self.teams == 1:
                self.score[goal_m[~ok], 1] += 1
            scorer = self.last_touch[goal_m]
            good = (scorer >= 0)
            own = good & (self.team_of[np.maximum(scorer, 0)] != scored[goal_m])
            self.goals_by[scorer[good & ~own]] += 1
            np.add.at(self.own_goals, (goal_m[own], self.team_of[scorer[own]]), 1)
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
            # Quietos: autos casi sin velocidad
            self.idle[:, t] += (self.world.speed[cars] < IDLE_SPEED).mean(axis=1)
            # Amontonados: pares de compañeros a menos de CROWD_DIST
            if self.N > 1:
                X, Y = self.world.x[cars], self.world.y[cars]
                close = 0
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        close = close + (np.hypot(X[:, i] - X[:, j], Y[:, i] - Y[:, j]) < CROWD_DIST)
                self.crowd[:, t] += close / (self.N * (self.N - 1) / 2)

    def team_breakdown(self, t):
        """Partes del puntaje del equipo t en cada partido (para el puntaje y para mostrarlo)"""
        cars = (np.arange(self.M)[:, None] * self.PPM + t * self.N + np.arange(self.N)[None, :])
        steps = max(1, self.steps)
        return {
            'goals': self.score[:, t].astype(np.float32),
            'conceded': (self.score[:, 1 - t] if self.teams == 2 else self.score[:, 1]).astype(np.float32),
            'own_goals': self.own_goals[:, t].astype(np.float32),
            'touches': np.minimum(self.touches[cars].sum(axis=1), 30).astype(np.float32),
            'good_kicks': np.minimum(self.good_kicks[cars].sum(axis=1), 15).astype(np.float32),
            'advance': np.maximum(self.advance[:, t], 0),
            'near': self.near[:, t] / steps,
            'crowd': self.crowd[:, t] / steps,
            'idle': self.idle[:, t] / steps,
        }

    def team_parts(self, t):
        """Puntos que aporta cada concepto al puntaje del equipo t (ya multiplicados por su peso)"""
        c, p = self.cfg, self.team_breakdown(t)
        return {
            'goals': p['goals'] * c.r_goal,
            'conceded': -p['conceded'] * c.p_conceded,
            'own_goals': -p['own_goals'] * c.p_own_goal,
            'touches': p['touches'] * c.r_touch,
            'good_kicks': p['good_kicks'] * c.r_kick,
            'advance': p['advance'] * c.r_advance,
            'near': p['near'] * c.r_near,
            'crowd': -p['crowd'] * c.p_crowd,
            'idle': -p['idle'] * c.p_idle,
        }

    def team_fitness(self):
        """(M, equipos) puntaje de cada equipo en su partido, con los pesos del agente"""
        fit = np.zeros((self.M, self.teams), dtype=np.float32)
        for t in range(self.teams):
            fit[:, t] = sum(self.team_parts(t).values())
        return fit


# ---------------------------------------------------------------------- #
# Escenario para el entrenador
# ---------------------------------------------------------------------- #
class FootballScenario:
    key = 'futbol'
    title = 'Fútbol'
    goal_text = "Meter la pelota en el arco rival"

    def __init__(self, field, team_size=1, opponent='bot_normal', match_steps=1500, ball_control='libre'):
        self.field = field
        self.ball_control = ball_control
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
        control = self.ball_control
        return [f"Rival: {self.opponent_label()}  ·  {self.team_size} por equipo", f"Cancha: {self.field.name}  ·  pelota {control}"]

    def draw_overlay(self, surf, to_screen, scale, font):
        pass

    def map_info(self):
        return {'map': self.field.name, 'map_file': self.field.file, 'team_size': self.team_size,
                'opponent': self.opponent, 'match_steps': self.match_steps, 'ball_control': self.ball_control}


# ---------------------------------------------------------------------- #
# Dibujo
# ---------------------------------------------------------------------- #
def draw_match(surf, sim, match, to_screen, scale, labels=None, highlight=None, stripes=None):
    """Dibuja los autos y la pelota de un partido. stripes: color de cada equipo en el techo (el del agente)"""
    from ai.sprites import draw_car
    w = sim.world
    font = pygame.font.Font(None, 22) if labels else None
    for k in range(sim.PPM):
        c = match * sim.PPM + k
        team = sim.team_of[c]
        cx, cy = to_screen((w.x[c], w.y[c]))
        stripe = stripes[team] if stripes else None
        draw_car(surf, (cx, cy), float(w.angle[c]), scale, TEAM_COLORS[team], stripe)
        if highlight is not None and c == highlight:
            pygame.draw.circle(surf, (255, 205, 40), (int(cx), int(cy)), int(max(12, 28 * scale)), 2)
        if labels and c in labels:
            t = font.render(labels[c], True, (255, 255, 255))
            surf.blit(t, t.get_rect(midbottom=(cx, cy - 20 * max(scale, 0.6))))
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
