"""
Escenarios de entrenamiento. Cada uno define:
- el mapa (dónde hay camino) y de dónde salen los autos
- cómo se mide el progreso de cada auto (su "puntaje" o fitness)
- cuándo un auto terminó (llegó, chocó, se quedó sin progresar)

PistaScenario: dar vueltas a una pista del editor pasando los checkpoints en orden.
LaberintoScenario: encontrar la salida de un laberinto generado al azar.

El progreso se mide con un "mapa de distancias": para cada punto del camino se
calcula (una sola vez, al empezar) cuánto falta para llegar al próximo objetivo
yendo por el camino. Así el puntaje sube a medida que el auto avanza de verdad,
y no por alejarse en línea recta.
"""

import math
import numpy as np
import pygame

import track_geometry as geo
from ai.world import surface_mask

CELL = geo.CELL

ROAD_DRAW = (205, 207, 214)
WALL_DRAW = (24, 26, 34)


def _grid(mask):
    return mask[CELL // 2::CELL, CELL // 2::CELL]


def _thick_cells(seg):
    cells = set()
    for cx, cy in geo._line_cells(*seg[:4]):
        cells.update({(cx, cy), (cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)})
    return cells


def _segments_cross(px, py, qx, qy, seg):
    """Cruce vectorizado: movimiento (p -> q) de cada auto contra su segmento seg (n, 4)"""
    ax, ay, bx, by = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]

    def orient(x1, y1, x2, y2, x3, y3):
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    d1 = orient(ax, ay, bx, by, px, py)
    d2 = orient(ax, ay, bx, by, qx, qy)
    d3 = orient(px, py, qx, qy, ax, ay)
    d4 = orient(px, py, qx, qy, bx, by)
    return ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))


class Scenario:
    key = ''
    title = ''
    goal_text = ''

    # Se completan en las subclases
    road_mask = None
    speed_mask = None
    slow_mask = None
    spawn = (0, 0, 0)
    background = None
    changed = False  # True cuando el mapa cambió (laberinto nuevo)

    @property
    def size(self):
        return self.road_mask.shape

    def start_generation(self, n, generation):
        """Prepara el estado por auto al comienzo de cada generación"""
        self.n = n
        self.progress = np.zeros(n, dtype=np.float32)
        self.best = np.full(n, -1e9, dtype=np.float32)
        self.last_improve = np.zeros(n, dtype=np.int32)
        self.finished_step = np.full(n, -1, dtype=np.int32)

    def _cell_values(self, field, x, y):
        cx = np.clip((x / CELL).astype(np.int32), 0, field.shape[0] - 1)
        cy = np.clip((y / CELL).astype(np.int32), 0, field.shape[1] - 1)
        return field[cx, cy]

    def fitness(self, cfg):
        """Progreso máximo alcanzado + premio por terminar rápido"""
        bonus = np.where(self.finished_step >= 0, (cfg.max_steps - self.finished_step) * cfg.r_fast, 0)
        return np.maximum(self.best, 0) + bonus

    def map_info(self):
        """Datos del mapa que se guardan junto al cerebro (para la galería y el modo Expo)"""
        return {}

    def mark_progress(self, idx, step, new_progress):
        self.progress[idx] = new_progress
        improved = new_progress > self.best[idx] + 2
        self.last_improve[idx[improved]] = step
        self.best[idx] = np.maximum(self.best[idx], new_progress)


# ---------------------------------------------------------------------- #
# Pista
# ---------------------------------------------------------------------- #
class PistaScenario(Scenario):
    key = 'pista'
    title = 'Pista'
    goal_text = "Dar vueltas pasando los checkpoints en orden"

    def __init__(self, track_data):
        road = track_data['track_layer']
        self.name = track_data.get('name', 'pista')
        self.file = track_data.get('file', self.name)
        self.road_surface = road
        self.road_mask = surface_mask(road)
        self.speed_mask = surface_mask(track_data['speed_zones'], (0, 255, 0)) if track_data.get('speed_zones') else None
        self.slow_mask = surface_mask(track_data['slow_zones'], (255, 255, 0)) if track_data.get('slow_zones') else None
        if self.speed_mask is not None and not self.speed_mask.any():
            self.speed_mask = None
        if self.slow_mask is not None and not self.slow_mask.any():
            self.slow_mask = None
        sp = track_data.get('spawn_point')
        self.spawn = tuple(sp[:3]) if sp else (road.get_width() / 2, road.get_height() * 0.77, -math.pi / 2)
        self.required_laps = int(track_data.get('required_laps', 1))

        checkpoints = [tuple(c[:4]) for c in track_data.get('checkpoints') or []]
        finish = tuple(track_data['finish_line'][:4]) if track_data.get('finish_line') else None
        self.auto_note = ''
        if not checkpoints:
            cps, auto_finish, _ = geo.auto_checkpoints(road, self.spawn)
            checkpoints = [tuple(c) for c in cps]
            if checkpoints:
                self.auto_note = "Checkpoints puestos automáticamente"
            if finish is None and auto_finish:
                finish = tuple(auto_finish)
        self.checkpoints = checkpoints
        self.finish = finish
        self.gates = checkpoints + ([finish] if finish else [])
        self.finish_idx = len(self.gates) - 1 if finish else -1
        self._build_fields()
        self.background = self._draw_background(track_data)

    # -------------------------------------------------------------- #
    def _build_fields(self):
        G = len(self.gates)
        self.gate_arr = np.array(self.gates, dtype=np.float32).reshape(-1, 4)
        grid = _grid(self.road_mask)
        if G < 2:
            # Sin suficientes checkpoints: el puntaje es la distancia recorrida
            self.fields = []
            self.start_gate, self.start_len = 0, 0.0
            return
        self.fields = []
        for g in range(G):
            sources = [c for c in geo._line_cells(*self.gates[g]) if 0 <= c[0] < grid.shape[0]
                       and 0 <= c[1] < grid.shape[1] and grid[c]]
            dist, _ = geo.bfs_field(grid, sources, _thick_cells(self.gates[g - 1]))
            self.fields.append(dist * CELL)

        # Largo de cada tramo: del checkpoint anterior a este
        self.seg_len = np.zeros(G, dtype=np.float32)
        for g in range(G):
            px1, py1, px2, py2 = self.gates[g - 1]
            mx, my = (px1 + px2) / 2, (py1 + py2) / 2
            vals = []
            for r in range(1, 6):
                for a in range(0, 360, 30):
                    v = self._field_at(self.fields[g], mx + math.cos(math.radians(a)) * r * CELL,
                                       my + math.sin(math.radians(a)) * r * CELL)
                    if v > 0:
                        vals.append(v)
            self.seg_len[g] = min(vals) if vals else 0
        self.lap_len = float(self.seg_len.sum())

        # Primer objetivo: el primer checkpoint que se cruza saliendo derecho desde la salida
        self.start_gate = 0
        best = None
        for g in range(G):
            d = self._field_at(self.fields[g], self.spawn[0], self.spawn[1])
            if d > 0 and (best is None or d < best):
                # Debe estar adelante: avanzar un poco tiene que acercar
                ahead = self._field_at(self.fields[g], self.spawn[0] + math.cos(self.spawn[2]) * 20,
                                       self.spawn[1] + math.sin(self.spawn[2]) * 20)
                if 0 < ahead < d:
                    best, self.start_gate = d, g
        self.start_len = best if best else self.seg_len[self.start_gate]

        # Sentido correcto de cruce de cada checkpoint: hacia donde el siguiente es alcanzable
        self.gate_dir = np.zeros((G, 2), dtype=np.float32)
        for g in range(G):
            x1, y1, x2, y2 = self.gates[g]
            nx, ny = -(y2 - y1), (x2 - x1)
            norm = math.hypot(nx, ny) or 1
            nx, ny = nx / norm, ny / norm
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            nxt = self.fields[(g + 1) % G]
            plus = self._field_at(nxt, mx + nx * 3 * CELL, my + ny * 3 * CELL)
            minus = self._field_at(nxt, mx - nx * 3 * CELL, my - ny * 3 * CELL)
            sign = 1 if (plus > 0 and (minus <= 0 or plus < minus)) else -1
            self.gate_dir[g] = (nx * sign, ny * sign)

    def _field_at(self, field, x, y):
        cx, cy = int(x / CELL), int(y / CELL)
        if 0 <= cx < field.shape[0] and 0 <= cy < field.shape[1]:
            return float(field[cx, cy])
        return -1.0

    def _draw_background(self, track_data):
        W, H = self.road_mask.shape
        bg = pygame.Surface((W, H))
        bg.fill(WALL_DRAW)
        road = pygame.surfarray.make_surface(np.where(self.road_mask[..., None], np.array(ROAD_DRAW), np.array(WALL_DRAW)).astype(np.uint8))
        bg.blit(road, (0, 0))
        for mask, color in ((self.speed_mask, (120, 220, 140)), (self.slow_mask, (235, 220, 110))):
            if mask is not None:
                layer = pygame.surfarray.make_surface((mask[..., None] * np.array(color)).astype(np.uint8))
                layer.set_colorkey((0, 0, 0))
                bg.blit(layer, (0, 0))
        return bg

    # -------------------------------------------------------------- #
    def start_generation(self, n, generation):
        super().start_generation(n, generation)
        self.next_gate = np.full(n, self.start_gate, dtype=np.int32)
        self.base = np.zeros(n, dtype=np.float32)
        self.cur_len = np.full(n, self.start_len if self.fields else 0, dtype=np.float32)
        self.passed = np.zeros(n, dtype=np.int32)
        self.laps = np.zeros(n, dtype=np.int32)
        self.distance = np.zeros(n, dtype=np.float32)

    def targets(self, idx):
        """Punto medio del próximo checkpoint (para la brújula)"""
        if not len(self.gates):
            return np.full(len(idx), self.spawn[0]), np.full(len(idx), self.spawn[1])
        g = self.gate_arr[self.next_gate[idx]]
        return (g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2

    def update(self, world, idx, prev_x, prev_y, step):
        """Actualiza el progreso. Devuelve qué autos de idx completaron la carrera."""
        x, y = world.x[idx], world.y[idx]
        done = np.zeros(len(idx), dtype=bool)
        if not self.fields:
            # Sin checkpoints: el puntaje es la distancia recorrida
            self.distance[idx] += world.speed[idx]
            self.mark_progress(idx, step, self.distance[idx])
            return done

        seg = self.gate_arr[self.next_gate[idx]]
        crossed = _segments_cross(prev_x, prev_y, x, y, seg)
        if crossed.any():
            d = self.gate_dir[self.next_gate[idx]]
            forward = (x - prev_x) * d[:, 0] + (y - prev_y) * d[:, 1] > 0
            ok = idx[crossed & forward]
            g = self.next_gate[ok]
            self.base[ok] += self.cur_len[ok]
            self.passed[ok] += 1
            lap = (g == self.finish_idx) & (self.passed[ok] >= len(self.gates))
            self.laps[ok[lap]] += 1
            self.next_gate[ok] = (g + 1) % len(self.gates)
            self.cur_len[ok] = self.seg_len[self.next_gate[ok]]
            if self.finish_idx < 0:
                # Sin meta: cada vuelta completa de checkpoints cuenta como vuelta
                self.laps[ok] = self.passed[ok] // len(self.gates)
            finished = ok[self.laps[ok] >= self.required_laps]
            self.finished_step[finished] = step
            done[np.isin(idx, finished)] = True

        # Distancia al próximo checkpoint según el mapa de distancias
        dist = np.empty(len(idx), dtype=np.float32)
        for g in np.unique(self.next_gate[idx]):
            sel = self.next_gate[idx] == g
            dist[sel] = self._cell_values(self.fields[g], x[sel], y[sel])
        valid = dist >= 0
        new = self.base[idx] + self.cur_len[idx] - dist
        new = np.where(valid, new, self.progress[idx])
        self.mark_progress(idx, step, new)
        return done

    def map_info(self):
        return {'map': self.name, 'map_file': self.file}

    def status(self):
        return [f"Vueltas para ganar: {self.required_laps}",
                f"Checkpoints: {len(self.checkpoints)}" + (f" ({self.auto_note})" if self.auto_note else "")]

    def leader_text(self, i):
        return f"Vuelta {min(self.laps[i] + 1, self.required_laps)}/{self.required_laps}"

    def draw_overlay(self, surf, to_screen, scale, font):
        for k, g in enumerate(self.gates):
            a, b = to_screen(g[:2]), to_screen(g[2:4])
            is_finish = k == self.finish_idx
            pygame.draw.line(surf, (230, 60, 60) if is_finish else (60, 120, 255), a, b, max(2, int(6 * scale)))
            if not is_finish:
                label = font.render(str(k + 1), True, (255, 255, 255))
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                r = label.get_rect(center=mid)
                pygame.draw.circle(surf, (60, 120, 255), r.center, max(r.w, r.h) // 2 + 3)
                surf.blit(label, r)


# ---------------------------------------------------------------------- #
# Laberinto
# ---------------------------------------------------------------------- #
MAZE_SIZES = {'chico': (9, 6), 'mediano': (13, 9), 'grande': (18, 12)}


def generate_maze(cols, rows, rng, braid=0.0):
    """Laberinto perfecto (backtracking) + atajos opcionales. Devuelve el set de pasajes."""
    visited = np.zeros((cols, rows), dtype=bool)
    passages = set()
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        cx, cy = stack[-1]
        options = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= cx + dx < cols and 0 <= cy + dy < rows and not visited[cx + dx, cy + dy]]
        if not options:
            stack.pop()
            continue
        nxt = options[rng.integers(len(options))]
        visited[nxt] = True
        passages.add(frozenset(((cx, cy), nxt)))
        stack.append(nxt)
    if braid > 0:
        # Abrir algunos callejones sin salida para crear caminos alternativos
        for cx in range(cols):
            for cy in range(rows):
                links = [p for p in passages if (cx, cy) in p]
                if len(links) == 1 and rng.random() < braid:
                    walls = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                             if 0 <= cx + dx < cols and 0 <= cy + dy < rows
                             and frozenset(((cx, cy), (cx + dx, cy + dy))) not in passages]
                    if walls:
                        passages.add(frozenset(((cx, cy), walls[rng.integers(len(walls))])))
    return passages


class LaberintoScenario(Scenario):
    key = 'laberinto'
    title = 'Laberinto'
    goal_text = "Encontrar la salida del laberinto"

    def __init__(self, size='mediano', new_every=0, braid=False, seed=None, W=1570, H=1080, maze_seed=None):
        self.size_name = size
        self.new_every = new_every
        self.braid = braid
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        self.maze_number = 0
        self.generate(maze_seed)

    def generate(self, maze_seed=None):
        """Nuevo laberinto. Con maze_seed se puede volver a armar exactamente el mismo."""
        self.maze_seed = int(self.rng.integers(1 << 30)) if maze_seed is None else int(maze_seed)
        maze_rng = np.random.default_rng(self.maze_seed)
        cols, rows = MAZE_SIZES[self.size_name]
        cell = min(self.W // cols, self.H // rows)
        ox, oy = (self.W - cell * cols) // 2, (self.H - cell * rows) // 2
        corridor = int(cell * 0.68)
        passages = generate_maze(cols, rows, maze_rng, 0.25 if self.braid else 0.0)

        surf = pygame.Surface((self.W, self.H))
        surf.fill((0, 0, 0))

        def center(c):
            return ox + c[0] * cell + cell // 2, oy + c[1] * cell + cell // 2
        for cx in range(cols):
            for cy in range(rows):
                x, y = center((cx, cy))
                pygame.draw.rect(surf, (255, 255, 255), (x - corridor // 2, y - corridor // 2, corridor, corridor))
        for p in passages:
            a, b = tuple(p)
            (x1, y1), (x2, y2) = center(a), center(b)
            rect = pygame.Rect(min(x1, x2) - corridor // 2, min(y1, y2) - corridor // 2,
                               abs(x2 - x1) + corridor, abs(y2 - y1) + corridor)
            pygame.draw.rect(surf, (255, 255, 255), rect)
        self.road_mask = surface_mask(surf)

        # Salida: esquina (0, 0). Meta: la celda más lejana por el laberinto
        neighbors = {}
        for p in passages:
            a, b = tuple(p)
            neighbors.setdefault(a, []).append(b)
            neighbors.setdefault(b, []).append(a)
        far, dist = (0, 0), {(0, 0): 0}
        queue = [(0, 0)]
        for c in queue:
            for nb in neighbors.get(c, []):
                if nb not in dist:
                    dist[nb] = dist[c] + 1
                    queue.append(nb)
                    if dist[nb] > dist[far]:
                        far = nb
        first = neighbors[(0, 0)][0]
        sx, sy = center((0, 0))
        fx, fy = center(first)
        self.spawn = (sx, sy, math.atan2(fy - sy, fx - sx))
        self.goal = center(far)
        self.goal_radius = corridor * 0.45

        grid = _grid(self.road_mask)
        gx, gy = int(self.goal[0] / CELL), int(self.goal[1] / CELL)
        field, _ = geo.bfs_field(grid, [(gx, gy)])
        self.field = field * CELL
        self.start_dist = float(self.field[int(sx / CELL), int(sy / CELL)])
        self.background = self._draw_background()
        self.maze_number += 1
        self.changed = True

    def _draw_background(self):
        bg = pygame.surfarray.make_surface(
            np.where(self.road_mask[..., None], np.array(ROAD_DRAW), np.array(WALL_DRAW)).astype(np.uint8))
        gx, gy = self.goal
        r = int(self.goal_radius)
        for k in range(4):
            for j in range(4):
                color = (30, 30, 30) if (k + j) % 2 else (250, 250, 250)
                pygame.draw.rect(bg, color, (gx - r + k * r // 2, gy - r + j * r // 2, r // 2 + 1, r // 2 + 1))
        return bg

    def start_generation(self, n, generation):
        if self.new_every and generation > 1 and (generation - 1) % self.new_every == 0:
            self.generate()
        super().start_generation(n, generation)
        self.reached = np.zeros(n, dtype=bool)

    def targets(self, idx):
        return np.full(len(idx), self.goal[0], dtype=np.float32), np.full(len(idx), self.goal[1], dtype=np.float32)

    def update(self, world, idx, prev_x, prev_y, step):
        x, y = world.x[idx], world.y[idx]
        d = self._cell_values(self.field, x, y)
        new = np.where(d >= 0, self.start_dist - d, self.progress[idx])
        self.mark_progress(idx, step, new)
        done = np.hypot(x - self.goal[0], y - self.goal[1]) < self.goal_radius
        if done.any():
            fin = idx[done]
            self.finished_step[fin] = step
            self.reached[fin] = True
            self.best[fin] = self.start_dist
        return done

    def map_info(self):
        return {'map': f"Laberinto {self.size_name}",
                'maze': {'size': self.size_name, 'new_every': self.new_every, 'braid': self.braid,
                         'seed': self.maze_seed}}

    def status(self):
        extra = f", nuevo cada {self.new_every} gen." if self.new_every else ""
        return [f"Tamaño: {self.size_name}{extra}", f"Laberinto n.º {self.maze_number}"]

    def leader_text(self, i):
        left = max(0, self.start_dist - self.progress[i])
        return f"Le faltan {left:.0f} px"

    def draw_overlay(self, surf, to_screen, scale, font):
        pass


# ---------------------------------------------------------------------- #
# Varias pistas seguidas
# ---------------------------------------------------------------------- #
class CurriculumScenario:
    """
    Entrena el mismo cerebro en varias pistas, una después de otra.
    Pasa a la siguiente pista cuando la mayoría la completa ('dominar') o cada N generaciones ('cada').
    Así el auto aprende a manejar en general y no se memoriza una sola pista.
    Todo lo que no está definido acá se lee de la pista actual.
    """
    key = 'pista'
    title = 'Varias pistas'
    goal_text = "Dominar cada pista y pasar a la siguiente"

    def __init__(self, scenarios, rule='dominar', every=10, threshold=50):
        self.scenarios = scenarios
        self.rule, self.every, self.threshold = rule, every, threshold
        self.index = 0
        self.since = 0         # generaciones en la pista actual
        self.rounds = 0        # veces que se completó la lista
        self.changed = True
        self.last_rate = 0.0

    @property
    def current(self):
        return self.scenarios[self.index]

    def __getattr__(self, name):
        # Solo se llama para atributos que no tiene el envoltorio
        return getattr(self.scenarios[self.__dict__['index']], name)

    @property
    def size(self):
        return self.current.size

    def start_generation(self, n, generation):
        if generation > 1 and hasattr(self.current, 'finished_step'):
            self.last_rate = float((self.current.finished_step >= 0).mean())
            self.since += 1
            move = (self.since >= self.every) if self.rule == 'cada' else (self.last_rate * 100 >= self.threshold)
            if move and len(self.scenarios) > 1:
                self.index = (self.index + 1) % len(self.scenarios)
                if self.index == 0:
                    self.rounds += 1
                self.since = 0
                self.changed = True
        self.current.start_generation(n, generation)

    def map_info(self):
        return {'map': f"{len(self.scenarios)} pistas", 'map_file': self.scenarios[0].file,
                'maps': [s.file for s in self.scenarios]}

    def status(self):
        rule = (f"cambia cada {self.every} gen." if self.rule == 'cada'
                else f"cambia cuando llega el {self.threshold} %")
        return [f"Pista {self.index + 1}/{len(self.scenarios)}: {self.current.name}",
                f"{rule} (gen. anterior: {self.last_rate * 100:.0f} %)"] + self.current.status()[:1]
