"""
Geometría para el editor de pistas:
- Curvas suaves (Catmull-Rom) para dibujar caminos
- Recorrido automático de la pista para ubicar checkpoints y meta
- Plantillas de pistas (oval, ocho, curvas en S, chicana)

Las funciones trabajan con una Surface de camino: negro = pared, cualquier otro color = camino.
"""

import math

WALL = (0, 0, 0)


# ---------------------------------------------------------------------- #
# Curvas
# ---------------------------------------------------------------------- #
def catmull_rom(points, closed=False, samples_per_segment=16):
    """Devuelve puntos de una curva suave que pasa por todos los puntos de control"""
    n = len(points)
    if n < 2:
        return list(points)
    if n == 2 and not closed:
        return list(points)

    def get(i):
        if closed:
            return points[i % n]
        return points[max(0, min(n - 1, i))]

    out = []
    segments = n if closed else n - 1
    for i in range(segments):
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        for s in range(samples_per_segment):
            t = s / samples_per_segment
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(points[0] if closed else points[-1])
    return out


# ---------------------------------------------------------------------- #
# Lectura de la pista
# ---------------------------------------------------------------------- #
def is_road(road, x, y):
    w, h = road.get_size()
    xi, yi = int(x), int(y)
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    return road.get_at((xi, yi))[:3] != WALL


def ray_length(road, x, y, angle, max_dist=400, step=3):
    dx, dy = math.cos(angle), math.sin(angle)
    d = 0
    while d < max_dist:
        d += step
        if not is_road(road, x + dx * d, y + dy * d):
            return d - step
    return max_dist


def cross_section(road, x, y, heading, max_half=300):
    """Segmento de borde a borde, perpendicular a heading, que pasa por (x, y)"""
    left = heading - math.pi / 2
    right = heading + math.pi / 2
    dl = ray_length(road, x, y, left, max_half, 2)
    dr = ray_length(road, x, y, right, max_half, 2)
    # Sobresalir un poco para que el auto no "pase por el costado"
    dl += 6
    dr += 6
    return (x + math.cos(left) * dl, y + math.sin(left) * dl,
            x + math.cos(right) * dr, y + math.sin(right) * dr)


# ---------------------------------------------------------------------- #
# Recorrido automático (BFS sobre una grilla del camino)
# ---------------------------------------------------------------------- #
CELL = 4  # Tamaño de celda de la grilla en píxeles


def road_mask(road):
    """Matriz booleana [x, y] con True donde hay camino"""
    import numpy as np
    import pygame
    arr = pygame.surfarray.pixels3d(road)
    mask = np.any(arr != 0, axis=2)
    del arr  # libera el lock de la Surface
    return mask


def _line_cells(x1, y1, x2, y2):
    """Celdas de la grilla que toca un segmento (en píxeles)"""
    n = int(max(abs(x2 - x1), abs(y2 - y1)) / (CELL / 2)) + 1
    cells = set()
    for k in range(n + 1):
        t = k / n
        cells.add((int((x1 + (x2 - x1) * t) / CELL), int((y1 + (y2 - y1) * t) / CELL)))
    return cells


def bfs_field(grid, sources, blocked=()):
    """
    Distancia (en celdas, con diagonales) desde las celdas origen a todo el camino.
    grid: matriz booleana [cx, cy]. Devuelve (dist, parent) con dist = -1 si no se llega.
    """
    import numpy as np
    from collections import deque
    gw, gh = grid.shape
    dist = np.full((gw, gh), -1.0, dtype=np.float32)
    parent = {}
    blocked = set(blocked)
    q = deque()
    for c in sources:
        if 0 <= c[0] < gw and 0 <= c[1] < gh and grid[c] and c not in blocked:
            dist[c] = 0
            q.append(c)
    steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
    while q:
        cx, cy = q.popleft()
        d = dist[cx, cy]
        for dx, dy, w in steps:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < gw and 0 <= ny < gh and grid[nx, ny] and dist[nx, ny] < 0 \
                    and (nx, ny) not in blocked:
                dist[nx, ny] = d + w
                parent[(nx, ny)] = (cx, cy)
                q.append((nx, ny))
    return dist, parent


def lap_field(road, spawn):
    """
    Corta la pista en la salida y mide, para cada celda, cuánto se avanzó en la vuelta.
    Devuelve (grid, dist, parent, lado_adelante, lado_atras, barrera) o None si la salida
    no está en el camino.
    """
    x, y, heading = spawn
    if not is_road(road, x, y):
        return None
    mask = road_mask(road)
    grid = mask[CELL // 2::CELL, CELL // 2::CELL]
    gw, gh = grid.shape
    # Barrera gruesa: con movimientos en diagonal una línea de 1 celda deja pasar
    barrier = set()
    for cx, cy in _line_cells(*cross_section(road, x, y, heading)):
        barrier.update({(cx, cy), (cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)})
    # Celdas de camino pegadas a la barrera, de cada lado
    hx, hy = math.cos(heading), math.sin(heading)
    ahead, behind = [], []
    for bx, by in barrier:
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                c = (bx + dx, by + dy)
                if c in barrier or not (0 <= c[0] < gw and 0 <= c[1] < gh) or not grid[c]:
                    continue
                side = ((c[0] + 0.5) * CELL - x) * hx + ((c[1] + 0.5) * CELL - y) * hy
                (ahead if side > 0 else behind).append(c)
    dist, parent = bfs_field(grid, ahead, barrier)
    return grid, dist, parent, set(ahead), set(behind), barrier


def steer_trace(road, x, y, heading, step=4, max_steps=8000):
    """
    Recorre la pista como un auto que siempre va hacia donde hay más espacio y se
    mantiene centrado. Sigue derecho en los cruces (sirve para pistas en forma de 8),
    pero puede fallar en curvas muy cerradas. Devuelve (camino, cerró_la_vuelta).
    """
    if not is_road(road, x, y):
        return [], False
    path = [(x, y, heading)]
    start = (x, y)
    traveled = 0.0
    max_turn = math.radians(9)
    for _ in range(max_steps):
        best_angle, best_score = heading, -1
        for deg in range(-70, 71, 7):
            a = heading + math.radians(deg)
            score = ray_length(road, x, y, a, 260, 4) - abs(deg) * 0.4
            if score > best_score:
                best_score, best_angle = score, a
        diff = (best_angle - heading + math.pi) % (2 * math.pi) - math.pi
        heading += max(-max_turn, min(max_turn, diff))
        dl = ray_length(road, x, y, heading - math.pi / 2, 200, 3)
        dr = ray_length(road, x, y, heading + math.pi / 2, 200, 3)
        shift = (dr - dl) * 0.15
        nx = x + math.cos(heading) * step + math.cos(heading + math.pi / 2) * shift
        ny = y + math.sin(heading) * step + math.sin(heading + math.pi / 2) * shift
        if not is_road(road, nx, ny):
            return path, False
        x, y = nx, ny
        traveled += step
        path.append((x, y, heading))
        if traveled > 400 and math.hypot(x - start[0], y - start[1]) < step * 6:
            return path, True
    return path, False


def trace_lap(road, x, y, heading):
    """
    Camino alrededor de la pista desde la salida. Devuelve ([(x, y, rumbo), ...], cerró_la_vuelta).
    Primero prueba manejando (sigue derecho en cruces); si no cierra, usa el camino más
    corto por la grilla, que no falla en curvas cerradas.
    Si la pista no es un circuito cerrado, devuelve el recorrido hasta el punto más lejano.
    """
    path, closed = steer_trace(road, x, y, heading)
    if closed:
        return path, True
    info = lap_field(road, (x, y, heading))
    if info is None:
        return [], False
    grid, dist, parent, ahead, behind, _ = info
    reached = [c for c in behind if dist[c] > 0]
    closed = bool(reached)
    if closed:
        end = min(reached, key=lambda c: dist[c])
    else:
        import numpy as np
        end = tuple(int(v) for v in np.unravel_index(np.argmax(dist), dist.shape))
    cells = [end]
    while cells[-1] not in ahead and cells[-1] in parent:
        cells.append(parent[cells[-1]])
    cells.reverse()
    pts = [((cx + 0.5) * CELL, (cy + 0.5) * CELL) for cx, cy in cells]
    path = []
    for i, (px, py) in enumerate(pts):
        a, b = pts[max(0, i - 5)], pts[min(len(pts) - 1, i + 5)]
        path.append((px, py, math.atan2(b[1] - a[1], b[0] - a[0])))
    return path, closed


def auto_checkpoints(road, spawn, spacing=380, skip_start=200):
    """
    Devuelve (checkpoints, meta_sugerida, ok). La meta va justo detrás de la salida,
    orientada para que el sentido correcto sea hacia donde mira el auto.
    """
    x, y, heading = spawn
    path, closed = trace_lap(road, x, y, heading)
    if len(path) < 20:
        return [], None, False

    # Largo acumulado del recorrido
    acc = [0.0]
    for a, b in zip(path, path[1:]):
        acc.append(acc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = acc[-1]
    count = max(3, min(12, int(total // spacing)))
    usable = total - 2 * skip_start
    checkpoints = []
    if usable > 0:
        j = 0
        for k in range(count):
            target = skip_start + usable * (k + 0.5) / count
            while j < len(acc) - 1 and acc[j] < target:
                j += 1
            px, py, ph = path[j]
            checkpoints.append(cross_section(road, px, py, ph))

    # Meta 40 px detrás de la salida
    bx = x - math.cos(heading) * 40
    by = y - math.sin(heading) * 40
    if not is_road(road, bx, by):
        bx, by = x, y
    fx1, fy1, fx2, fy2 = cross_section(road, bx, by, heading)
    # cross_section devuelve (izquierda -> derecha). Con esa orientación la dirección
    # rotada +90° apunta hacia atrás del auto, por eso se invierte.
    finish = (fx2, fy2, fx1, fy1)
    return checkpoints, finish, closed


# ---------------------------------------------------------------------- #
# Plantillas
# ---------------------------------------------------------------------- #
def template_points(name, W, H):
    """Puntos de control (curva cerrada) y ancho del camino para cada plantilla"""
    cx, cy = W / 2, H / 2
    if name == 'oval':
        pts = [(cx + W * 0.36 * math.cos(t), cy + H * 0.34 * math.sin(t))
               for t in [i * 2 * math.pi / 12 for i in range(12)]]
        return pts, 120
    if name == 'ocho':
        pts = []
        for i in range(24):
            t = math.pi / 2 + i * 2 * math.pi / 24  # Empieza en un extremo, lejos del cruce
            pts.append((cx + W * 0.38 * math.sin(t), cy + H * 0.62 * math.sin(t) * math.cos(t)))
        return pts, 90
    if name == 'curvas':
        pts = [(0.12, 0.2), (0.35, 0.13), (0.55, 0.3), (0.75, 0.13), (0.9, 0.3),
               (0.88, 0.6), (0.7, 0.85), (0.5, 0.68), (0.3, 0.88), (0.1, 0.7), (0.15, 0.45)]
        return [(x * W, y * H) for x, y in pts], 95
    if name == 'chicana':
        pts = [(0.12, 0.15), (0.5, 0.15), (0.88, 0.15), (0.92, 0.5), (0.88, 0.85),
               (0.72, 0.85), (0.64, 0.72), (0.56, 0.85), (0.44, 0.85), (0.36, 0.72),
               (0.28, 0.85), (0.12, 0.85), (0.08, 0.5)]
        return [(x * W, y * H) for x, y in pts], 95
    raise ValueError(name)


def template_spawn(points):
    """Salida en el medio del primer tramo, mirando hacia el segundo punto"""
    curve = catmull_rom(points, closed=True)
    i = len(curve) // (2 * len(points)) + 2
    (x1, y1), (x2, y2) = curve[i], curve[i + 1]
    return (x1, y1, math.atan2(y2 - y1, x2 - x1))
