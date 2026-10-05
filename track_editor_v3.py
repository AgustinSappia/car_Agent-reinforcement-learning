"""
Track Editor V3 - Editor de pistas por herramientas

Diferencias con V2:
- Una sola herramienta activa a la vez (Camino, Goma, Zonas, Salida, Meta, Checkpoint, Seleccionar)
- Salida, meta y checkpoints son OBJETOS (coordenadas), no dibujos: se pueden
  seleccionar, mover, invertir y borrar uno por uno
- Deshacer / Rehacer (Ctrl+Z / Ctrl+Y)
- Lienzo fijo del tamaño que usa el juego (1570x1080), escalado para entrar en cualquier pantalla
- Trazos continuos aunque el mouse se mueva rápido
- Ayuda y lista de verificación en pantalla
- Guarda en el mismo formato que V2 (track_loader / train_genetic2 no cambian)
- Herramienta Curva, checkpoints automáticos, plantillas, zoom, simetría y modo Probar
"""

import pygame
import math
import json
import os
import zlib
from datetime import datetime

import track_geometry as geo

# Tamaño lógico de la pista (lo que usa el entrenamiento)
TRACK_W, TRACK_H = 1570, 1080

# Colores que lee el juego (no cambiar: environment.py compara colores exactos)
ROAD_COLOR = (255, 255, 255)
WALL_COLOR = (0, 0, 0)
SPEED_ZONE_COLOR = (0, 255, 0)
SLOW_ZONE_COLOR = (255, 255, 0)
FINISH_LINE_COLOR = (255, 0, 0)
CHECKPOINT_COLOR = (0, 100, 255)

# Colores de interfaz
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG = (24, 26, 34)
PANEL = (36, 39, 50)
PANEL_LIGHT = (52, 56, 70)
TEXT_DIM = (150, 155, 170)
YELLOW = (255, 205, 40)
GREEN = (60, 200, 110)
RED = (230, 70, 70)
ORANGE = (255, 150, 40)
CYAN = (70, 210, 230)
BLUE = (60, 130, 255)

TOOLBAR_W = 250
STATUS_H = 34

# (id, nombre, tecla, color de muestra)
TOOLS = [
    ('road', "Camino", '1', ROAD_COLOR),
    ('curve', "Curva", '8', (170, 170, 255)),
    ('erase', "Goma", '2', (90, 90, 90)),
    ('speed', "Zona rápida", '3', SPEED_ZONE_COLOR),
    ('slow', "Zona lenta", '4', SLOW_ZONE_COLOR),
    ('spawn', "Salida", '5', CYAN),
    ('finish', "Meta", '6', FINISH_LINE_COLOR),
    ('checkpoint', "Checkpoint", '7', CHECKPOINT_COLOR),
    ('select', "Seleccionar", 'V', ORANGE),
]
TOOL_KEYS = {pygame.K_1: 'road', pygame.K_2: 'erase', pygame.K_3: 'speed', pygame.K_4: 'slow',
             pygame.K_5: 'spawn', pygame.K_6: 'finish', pygame.K_7: 'checkpoint', pygame.K_8: 'curve', pygame.K_v: 'select'}

TOOL_HELP = {
    'road': "Clic y arrastrá para dibujar el camino. Clic derecho borra. Rueda / [ ]: tamaño del pincel.",
    'curve': "Clic para agregar puntos. Clic en el primer punto cierra la vuelta. Enter o clic derecho: terminar. Retroceso: borrar punto.",
    'erase': "Clic y arrastrá para borrar camino (vuelve a ser pared). Rueda / [ ]: tamaño.",
    'speed': "Pintá zonas donde el auto acelera. Clic derecho borra la zona.",
    'slow': "Pintá zonas donde el auto frena. Clic derecho borra la zona.",
    'spawn': "Clic donde arranca el auto y arrastrá hacia donde mira.",
    'finish': "Clic y arrastrá de un borde del camino al otro. La flecha verde es el sentido de carrera. I: invertir.",
    'checkpoint': "Clic y arrastrá de borde a borde para agregar un checkpoint. Los autos deben pasar todos.",
    'select': "Clic en un objeto para seleccionarlo, arrastrá para moverlo o sus puntas. Supr: borrar. I: invertir meta.",
}

TEMPLATES = [('oval', "Oval"), ('ocho', "Ocho"), ('curvas', "Curvas en S"), ('chicana', "Chicana")]
SYMMETRY_LABELS = ["Simetría: no", "Simetría: izq-der", "Simetría: arr-abj", "Simetría: 4 lados"]

MAX_HISTORY = 40
PICK_RADIUS = 14  # en píxeles de pantalla


def dist_point_segment(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def finish_direction(line):
    """Sentido correcto de cruce de la meta.

    train_genetic2 / environment aceptan el cruce cuando cross(linea, movimiento) > 0,
    o sea cuando el auto se mueve en la dirección de la línea rotada +90°.
    """
    x1, y1, x2, y2 = line[:4]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1
    return (-dy / length, dx / length)


class TrackEditorV3:
    def __init__(self, track_name=None):
        pygame.init()
        info = pygame.display.Info()
        # Ventana que entre en la pantalla (deja margen para la barra de tareas)
        max_w = (info.current_w - 60) if info.current_w > 0 else 1600
        max_h = (info.current_h - 120) if info.current_h > 0 else 900
        self.width = max(1000, min(1700, max_w))
        self.height = max(640, min(1000, max_h))
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Editor de Pistas")
        self.clock = pygame.time.Clock()
        pygame.key.set_repeat(300, 50)

        self.title_font = pygame.font.Font(None, 34)
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)
        self.tiny_font = pygame.font.Font(None, 19)

        # Viewport del lienzo
        avail_w = self.width - TOOLBAR_W - 20
        avail_h = self.height - STATUS_H - 20
        self.base_scale = min(avail_w / TRACK_W, avail_h / TRACK_H)
        self.zoom = 1.0
        self.offset = [0.0, 0.0]  # Esquina superior izquierda visible (coordenadas de pista)
        view_w, view_h = int(TRACK_W * self.base_scale), int(TRACK_H * self.base_scale)
        self.view = pygame.Rect(TOOLBAR_W + 10 + (avail_w - view_w) // 2,
                                10 + (avail_h - view_h) // 2, view_w, view_h)

        # Capas pintadas
        self.road = pygame.Surface((TRACK_W, TRACK_H))
        self.road.fill(WALL_COLOR)
        self.speed = self._zone_surface()
        self.slow = self._zone_surface()

        # Objetos
        self.spawn = None          # (x, y, angulo)
        self.finish = None         # (x1, y1, x2, y2)
        self.checkpoints = []      # [(x1, y1, x2, y2), ...]
        self.required_laps = 3

        # Estado de edición
        self.tool = 'road'
        self.brush = 30
        self.drag = None           # Lo que se está arrastrando ahora
        self.selected = None       # ('spawn',) | ('finish',) | ('checkpoint', i)
        self.undo_stack = []
        self.redo_stack = []
        self.dirty = False
        self.view_dirty = True
        self.check_dirty = True
        self.exit_after_save = False
        self.view_cache = None
        self.show_grid = False
        self.modal = None          # None | 'save_name' | 'unsaved'
        self.name_text = ''
        self.toast = None
        self.running = True
        self.curve_points = []
        self.symmetry = 0          # 0 no, 1 horizontal, 2 vertical, 3 ambas
        self.test = None           # TestDrive activo
        self.pan = None            # (mouse_inicio, offset_inicio)
        self.composed = None
        self.compose_dirty = True

        self.track_name = None     # Nombre de archivo (track_YYYYmmdd_HHMMSS)
        self.display_name = None
        self.created = None
        if track_name:
            self.load_track(track_name)
        self.update_caption()

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    @staticmethod
    def _zone_surface():
        s = pygame.Surface((TRACK_W, TRACK_H))
        s.fill(BLACK)
        s.set_colorkey(BLACK)
        return s

    def update_caption(self):
        name = self.display_name or self.track_name or "Pista nueva"
        pygame.display.set_caption(f"Editor de Pistas - {name}{' *' if self.dirty else ''}")

    @property
    def scale(self):
        return self.base_scale * self.zoom

    def to_canvas(self, pos):
        return ((pos[0] - self.view.x) / self.scale + self.offset[0],
                (pos[1] - self.view.y) / self.scale + self.offset[1])

    def to_screen(self, pt):
        return (self.view.x + (pt[0] - self.offset[0]) * self.scale,
                self.view.y + (pt[1] - self.offset[1]) * self.scale)

    def clamp_offset(self):
        vis_w, vis_h = self.view.w / self.scale, self.view.h / self.scale
        self.offset[0] = max(0.0, min(TRACK_W - vis_w, self.offset[0]))
        self.offset[1] = max(0.0, min(TRACK_H - vis_h, self.offset[1]))

    def zoom_at(self, pos, factor):
        """Zoom manteniendo fijo el punto bajo el mouse"""
        before = self.to_canvas(pos)
        self.zoom = max(1.0, min(5.0, self.zoom * factor))
        after = self.to_canvas(pos)
        self.offset[0] += before[0] - after[0]
        self.offset[1] += before[1] - after[1]
        self.clamp_offset()
        self.view_dirty = True

    def reset_zoom(self):
        self.zoom = 1.0
        self.offset = [0.0, 0.0]
        self.view_dirty = True

    def clamp(self, pt):
        return (max(0, min(TRACK_W - 1, pt[0])), max(0, min(TRACK_H - 1, pt[1])))

    def show_toast(self, text, color=GREEN):
        self.toast = (text, color, pygame.time.get_ticks() + 3000)

    def mark_changed(self):
        self.dirty = True
        self.view_dirty = True
        self.compose_dirty = True
        self.check_dirty = True
        self.update_caption()

    # ------------------------------------------------------------------ #
    # Deshacer / Rehacer
    # ------------------------------------------------------------------ #
    def _layer(self, name):
        return {'road': self.road, 'speed': self.speed, 'slow': self.slow}[name]

    def snapshot(self, layers=()):
        """Guarda el estado actual. Las capas se guardan comprimidas (son casi todo negro)."""
        if isinstance(layers, str):
            layers = (layers,)
        return {
            'objects': (self.spawn, self.finish, list(self.checkpoints), self.required_laps),
            'layers': [(name, zlib.compress(pygame.image.tobytes(self._layer(name), 'RGB'), 1))
                       for name in layers],
        }

    def push_undo(self, layers=()):
        self.undo_stack.append(self.snapshot(layers))
        if len(self.undo_stack) > MAX_HISTORY:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def restore(self, state):
        self.spawn, self.finish, cps, self.required_laps = state['objects']
        self.checkpoints = list(cps)
        for name, data in state['layers']:
            surf = pygame.image.frombytes(zlib.decompress(data), (TRACK_W, TRACK_H), 'RGB')
            self._layer(name).blit(surf, (0, 0))
        self.selected = None
        self.mark_changed()

    def undo(self):
        if not self.undo_stack:
            return
        state = self.undo_stack.pop()
        self.redo_stack.append(self.snapshot([n for n, _ in state['layers']]))
        self.restore(state)

    def redo(self):
        if not self.redo_stack:
            return
        state = self.redo_stack.pop()
        self.undo_stack.append(self.snapshot([n for n, _ in state['layers']]))
        self.restore(state)

    # ------------------------------------------------------------------ #
    # Pintar
    # ------------------------------------------------------------------ #
    def paint_target(self, erase):
        """(capa, color) según la herramienta actual"""
        if self.tool == 'road':
            return 'road', WALL_COLOR if erase else ROAD_COLOR
        if self.tool == 'erase':
            return 'road', WALL_COLOR
        if self.tool == 'speed':
            return 'speed', BLACK if erase else SPEED_ZONE_COLOR
        if self.tool == 'slow':
            return 'slow', BLACK if erase else SLOW_ZONE_COLOR
        return None, None

    def mirrors(self, pt):
        """El punto y sus reflejos según la simetría activa"""
        x, y = pt
        pts = [(x, y)]
        if self.symmetry in (1, 3):
            pts.append((TRACK_W - x, y))
        if self.symmetry in (2, 3):
            pts.append((x, TRACK_H - y))
        if self.symmetry == 3:
            pts.append((TRACK_W - x, TRACK_H - y))
        return pts

    def stroke(self, layer, color, a, b):
        for ma, mb in zip(self.mirrors(a), self.mirrors(b)):
            self._stroke(layer, color, ma, mb)

    def _stroke(self, layer, color, a, b):
        """Trazo continuo de a hasta b (círculos en los extremos + línea gruesa)"""
        surf = self._layer(layer)
        r = self.brush
        pygame.draw.circle(surf, color, (int(b[0]), int(b[1])), r)
        if a != b:
            pygame.draw.circle(surf, color, (int(a[0]), int(a[1])), r)
            pygame.draw.line(surf, color, a, b, r * 2)
            # Rellenar huecos de la línea gruesa en diagonales
            steps = int(math.hypot(b[0] - a[0], b[1] - a[1]) // max(1, r // 2))
            for k in range(1, steps):
                t = k / steps
                pygame.draw.circle(surf, color, (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t)), r)
        self.view_dirty = True
        self.check_dirty = True
        self.compose_dirty = True

    # ------------------------------------------------------------------ #
    # Objetos: selección
    # ------------------------------------------------------------------ #
    def objects(self):
        """Lista de (clave, segmento o punto) para hit-testing"""
        items = []
        if self.spawn:
            items.append((('spawn',), self.spawn))
        if self.finish:
            items.append((('finish',), self.finish))
        for i, cp in enumerate(self.checkpoints):
            items.append((('checkpoint', i), cp))
        return items

    def get_object(self, key):
        if key[0] == 'spawn':
            return self.spawn
        if key[0] == 'finish':
            return self.finish
        return self.checkpoints[key[1]]

    def set_object(self, key, value):
        if key[0] == 'spawn':
            self.spawn = value
        elif key[0] == 'finish':
            self.finish = value
        else:
            self.checkpoints[key[1]] = value
        self.mark_changed()

    def hit_test(self, cpos):
        """Devuelve (clave, parte) bajo el mouse. parte: 'p1' | 'p2' | 'body'"""
        tol = PICK_RADIUS / self.scale
        best = None
        for key, obj in self.objects():
            if key[0] == 'spawn':
                d = math.hypot(cpos[0] - obj[0], cpos[1] - obj[1])
                if d < tol * 1.6:
                    return key, 'body'
                continue
            p1, p2 = obj[:2], obj[2:4]
            if math.hypot(cpos[0] - p1[0], cpos[1] - p1[1]) < tol:
                return key, 'p1'
            if math.hypot(cpos[0] - p2[0], cpos[1] - p2[1]) < tol:
                return key, 'p2'
            d = dist_point_segment(cpos, p1, p2)
            if d < tol and (best is None or d < best[0]):
                best = (d, key)
        return (best[1], 'body') if best else (None, None)

    def delete_selected(self):
        if not self.selected:
            return
        self.push_undo()
        kind = self.selected[0]
        if kind == 'spawn':
            self.spawn = None
        elif kind == 'finish':
            self.finish = None
        else:
            self.checkpoints.pop(self.selected[1])
        self.selected = None
        self.mark_changed()

    def invert_finish(self):
        if not self.finish:
            return
        self.push_undo()
        x1, y1, x2, y2 = self.finish[:4]
        self.finish = (x2, y2, x1, y1)
        self.mark_changed()
        self.show_toast("Sentido de la meta invertido")

    # ------------------------------------------------------------------ #
    # Mouse
    # ------------------------------------------------------------------ #
    def on_mouse_down(self, pos, button):
        if not self.view.collidepoint(pos):
            return
        keys = pygame.key.get_pressed()
        if button == 2 or (button == 1 and keys[pygame.K_SPACE]):
            self.pan = (pos, list(self.offset))
            return
        c = self.clamp(self.to_canvas(pos))
        if self.tool == 'curve':
            if button == 3:
                self.finish_curve(closed=False)
            elif button == 1:
                if len(self.curve_points) >= 3:
                    first = self.to_screen(self.curve_points[0])
                    if math.hypot(pos[0] - first[0], pos[1] - first[1]) < PICK_RADIUS:
                        self.finish_curve(closed=True)
                        return
                self.curve_points.append(c)
            return
        layer, color = self.paint_target(erase=(button == 3))

        if layer and button in (1, 3):
            self.push_undo(layer)
            self.stroke(layer, color, c, c)
            self.drag = ('paint', layer, color, c)
            self.mark_changed()
        elif button != 1:
            return
        elif self.tool == 'spawn':
            self.push_undo()
            angle = self.spawn[2] if self.spawn else -math.pi / 2
            self.spawn = (c[0], c[1], angle)
            self.drag = ('spawn_aim', c)
            self.selected = ('spawn',)
            self.mark_changed()
        elif self.tool in ('finish', 'checkpoint'):
            self.drag = ('line', self.tool, c, c)
        elif self.tool == 'select':
            key, part = self.hit_test(c)
            self.selected = key
            if key:
                self.push_undo()
                self.drag = ('move', key, part, c, self.get_object(key))

    def on_mouse_move(self, pos):
        if self.pan:
            (sx, sy), (ox, oy) = self.pan
            self.offset = [ox - (pos[0] - sx) / self.scale, oy - (pos[1] - sy) / self.scale]
            self.clamp_offset()
            self.view_dirty = True
            return
        if not self.drag:
            return
        c = self.clamp(self.to_canvas(pos))
        kind = self.drag[0]
        if kind == 'paint':
            _, layer, color, last = self.drag
            self.stroke(layer, color, last, c)
            self.drag = ('paint', layer, color, c)
        elif kind == 'spawn_aim':
            origin = self.drag[1]
            if math.hypot(c[0] - origin[0], c[1] - origin[1]) > 5:
                self.spawn = (origin[0], origin[1], math.atan2(c[1] - origin[1], c[0] - origin[0]))
                self.mark_changed()
        elif kind == 'line':
            self.drag = ('line', self.drag[1], self.drag[2], c)
        elif kind == 'move':
            _, key, part, start, orig = self.drag
            dx, dy = c[0] - start[0], c[1] - start[1]
            if key[0] == 'spawn':
                self.set_object(key, (orig[0] + dx, orig[1] + dy, orig[2]))
            elif part == 'p1':
                self.set_object(key, (c[0], c[1], orig[2], orig[3]))
            elif part == 'p2':
                self.set_object(key, (orig[0], orig[1], c[0], c[1]))
            else:
                self.set_object(key, (orig[0] + dx, orig[1] + dy, orig[2] + dx, orig[3] + dy))

    def on_mouse_up(self, pos):
        if self.pan:
            self.pan = None
            return
        if not self.drag:
            return
        if self.drag[0] == 'line':
            _, tool, a, b = self.drag
            if math.hypot(b[0] - a[0], b[1] - a[1]) * self.scale > 8:
                self.push_undo()
                line = (a[0], a[1], b[0], b[1])
                if tool == 'finish':
                    self.finish = line
                    self.selected = ('finish',)
                else:
                    self.checkpoints.append(line)
                    self.selected = ('checkpoint', len(self.checkpoints) - 1)
                self.mark_changed()
        elif self.drag[0] == 'move':
            # Si no se movió nada, sacar el snapshot que no hizo falta
            _, key, _, _, orig = self.drag
            if self.get_object(key) == orig and self.undo_stack:
                self.undo_stack.pop()
        self.drag = None


    # ------------------------------------------------------------------ #
    # Curva, checkpoints automáticos, plantillas, prueba
    # ------------------------------------------------------------------ #
    def finish_curve(self, closed):
        pts = self.curve_points
        self.curve_points = []
        if len(pts) < 2:
            return
        self.push_undo('road')
        line = geo.catmull_rom(pts, closed=closed)
        for a, b in zip(line, line[1:]):
            self.stroke('road', ROAD_COLOR, a, b)
        self.mark_changed()
        self.show_toast("Curva cerrada" if closed else "Curva agregada")

    def auto_checkpoints(self):
        if not self.spawn or not self.road.get_at((int(self.spawn[0]), int(self.spawn[1])))[:3] != WALL_COLOR:
            self.show_toast("Primero poné la salida sobre el camino", ORANGE)
            return
        cps, finish, closed = geo.auto_checkpoints(self.road, self.spawn)
        if not cps:
            self.show_toast("No pude recorrer la pista desde la salida", RED)
            return
        self.push_undo()
        self.checkpoints = cps
        if not self.finish:
            self.finish = finish
        self.selected = None
        self.mark_changed()
        if closed:
            self.show_toast(f"{len(cps)} checkpoints ubicados")
        else:
            self.show_toast(f"{len(cps)} checkpoints, pero la vuelta no cierra: revisalos", ORANGE)

    def apply_template(self, name):
        self.clear_all(toast=False)
        pts, width = geo.template_points(name, TRACK_W, TRACK_H)
        brush, symmetry = self.brush, self.symmetry
        self.brush, self.symmetry = width // 2, 0
        line = geo.catmull_rom(pts, closed=True)
        for a, b in zip(line, line[1:]):
            self._stroke('road', ROAD_COLOR, a, b)
        self.brush, self.symmetry = brush, symmetry
        self.spawn = geo.template_spawn(pts)
        cps, finish, _ = geo.auto_checkpoints(self.road, self.spawn)
        self.checkpoints, self.finish = cps, finish
        self.mark_changed()
        self.show_toast(f"Plantilla {dict(TEMPLATES)[name]} creada (Ctrl+Z para volver)")

    def start_test(self):
        if not self.spawn or self.road.get_at((int(self.spawn[0]), int(self.spawn[1])))[:3] == WALL_COLOR:
            self.show_toast("Para probar, poné la salida sobre el camino", ORANGE)
            return
        from track_test_drive import TestDrive
        self.curve_points = []
        self.drag = None
        self.test = TestDrive(self.road, self.speed, self.slow, self.spawn, self.finish,
                              self.checkpoints, self.required_laps)

    # ------------------------------------------------------------------ #
    # Toolbar
    # ------------------------------------------------------------------ #
    def toolbar_items(self):
        """Botones de la barra lateral: (id, texto, rect, activo)"""
        items = []
        col_w = (TOOLBAR_W - 30) // 2
        y = 54
        for k, (tool_id, label, key, _) in enumerate(TOOLS):
            x = 12 + (k % 2) * (col_w + 6)
            items.append(('tool:' + tool_id, label, pygame.Rect(x, y + (k // 2) * 34, col_w, 30),
                          self.tool == tool_id))
        y += ((len(TOOLS) + 1) // 2) * 34 + 2
        self.brush_y = y
        items.append(('brush-', "-", pygame.Rect(12, y + 16, 36, 26), False))
        items.append(('brush+', "+", pygame.Rect(TOOLBAR_W - 48, y + 16, 36, 26), False))
        y += 46
        self.laps_y = y
        items.append(('laps-', "-", pygame.Rect(12, y + 16, 36, 26), False))
        items.append(('laps+', "+", pygame.Rect(TOOLBAR_W - 48, y + 16, 36, 26), False))
        y += 50
        grid = [('undo', "Deshacer"), ('redo', "Rehacer"),
                ('templates', "Plantillas (T)"), ('autocp', "Checkp. auto (A)"),
                ('symmetry', SYMMETRY_LABELS[self.symmetry]), ('clear', "Borrar todo")]
        for k, (bid, label) in enumerate(grid):
            x = 12 + (k % 2) * (col_w + 6)
            items.append((bid, label, pygame.Rect(x, y + (k // 2) * 34, col_w, 30), bid == 'symmetry' and self.symmetry > 0))
        y += 3 * 34 + 4
        items.append(('test', "Probar (P)", pygame.Rect(12, y, TOOLBAR_W - 24, 32), False))
        y += 38
        items.append(('save', "Guardar", pygame.Rect(12, y, col_w, 32), False))
        items.append(('exit', "Salir (Esc)", pygame.Rect(18 + col_w, y, col_w, 32), False))
        return items

    def on_toolbar_click(self, pos):
        for bid, _, rect, _ in self.toolbar_items():
            if rect.collidepoint(pos):
                self.do_action(bid)
                return True
        return pos[0] < TOOLBAR_W

    def do_action(self, bid):
        if bid.startswith('tool:'):
            if self.tool == 'curve' and bid != 'tool:curve':
                self.curve_points = []
            self.tool = bid[5:]
            self.drag = None
        elif bid == 'brush-':
            self.brush = max(5, self.brush - 5)
        elif bid == 'brush+':
            self.brush = min(120, self.brush + 5)
        elif bid in ('laps-', 'laps+'):
            new = max(1, min(10, self.required_laps + (1 if bid == 'laps+' else -1)))
            if new != self.required_laps:
                self.push_undo()
                self.required_laps = new
                self.mark_changed()
        elif bid == 'undo':
            self.undo()
        elif bid == 'redo':
            self.redo()
        elif bid == 'templates':
            self.modal = 'templates'
        elif bid == 'autocp':
            self.auto_checkpoints()
        elif bid == 'symmetry':
            self.symmetry = (self.symmetry + 1) % 4
        elif bid == 'test':
            self.start_test()
        elif bid == 'clear':
            self.clear_all()
        elif bid == 'save':
            self.request_save()
        elif bid == 'exit':
            self.request_exit()

    def clear_all(self, toast=True):
        self.push_undo(('road', 'speed', 'slow'))
        self.road.fill(WALL_COLOR)
        self.speed.fill(BLACK)
        self.slow.fill(BLACK)
        self.spawn, self.finish, self.checkpoints = None, None, []
        self.selected = None
        self.mark_changed()
        if toast:
            self.show_toast("Pista borrada (Ctrl+Z para deshacer)", ORANGE)

    # ------------------------------------------------------------------ #
    # Verificación
    # ------------------------------------------------------------------ #
    def checklist(self):
        """[(texto, ok)] de lo necesario para entrenar"""
        if self.check_dirty:
            self._road_ok = pygame.transform.average_color(self.road)[:3] != WALL_COLOR
            self.check_dirty = False
        road_ok = self._road_ok
        spawn_on_road = False
        if self.spawn:
            x, y = int(self.spawn[0]), int(self.spawn[1])
            spawn_on_road = self.road.get_at((x, y))[:3] != WALL_COLOR
        return [
            ("Camino dibujado", road_ok),
            ("Salida sobre el camino", spawn_on_road),
            ("Línea de meta", self.finish is not None),
            (f"Checkpoints ({len(self.checkpoints)})", len(self.checkpoints) > 0),
        ]

    # ------------------------------------------------------------------ #
    # Guardar / cargar (mismo formato que V2)
    # ------------------------------------------------------------------ #
    def request_save(self):
        if self.track_name:
            self.save_track()
        else:
            self.modal = 'save_name'
            self.name_text = self.display_name or ''

    def request_exit(self):
        if self.dirty:
            self.modal = 'unsaved'
        else:
            self.running = False

    def save_track(self):
        os.makedirs('tracks', exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.track_name:
            self.track_name = f"track_{stamp}"
            self.created = stamp
        name = self.track_name
        base = os.path.join('tracks', name)

        finish_layer = pygame.Surface((TRACK_W, TRACK_H))
        finish_layer.fill(BLACK)
        checkpoint_layer = pygame.Surface((TRACK_W, TRACK_H))
        checkpoint_layer.fill(BLACK)
        if self.finish:
            pygame.draw.line(finish_layer, FINISH_LINE_COLOR, self.finish[:2], self.finish[2:4], 10)
        for cp in self.checkpoints:
            pygame.draw.line(checkpoint_layer, CHECKPOINT_COLOR, cp[:2], cp[2:4], 8)

        combined = self.road.copy()
        for layer in (self.speed, self.slow, checkpoint_layer, finish_layer):
            combined.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)

        pygame.image.save(combined, f"{base}.png")
        pygame.image.save(self.road, f"{base}_track.png")
        pygame.image.save(finish_layer, f"{base}_finish.png")
        pygame.image.save(checkpoint_layer, f"{base}_checkpoint.png")
        pygame.image.save(self.speed, f"{base}_speed.png")
        pygame.image.save(self.slow, f"{base}_slow.png")
        pygame.image.save(pygame.transform.smoothscale(combined, (200, 150)), f"{base}_thumb.png")

        finish = None
        if self.finish:
            x1, y1, x2, y2 = (round(v) for v in self.finish[:4])
            finish = [x1, y1, x2, y2, math.atan2(y2 - y1, x2 - x1)]
        metadata = {
            'name': name,
            'width': TRACK_W,
            'height': TRACK_H,
            'spawn_point': [round(self.spawn[0]), round(self.spawn[1]), self.spawn[2]] if self.spawn else None,
            'finish_line': finish,
            'checkpoints': [[round(v) for v in cp[:4]] for cp in self.checkpoints],
            'required_laps': self.required_laps,
            'created': self.created or stamp,
            'modified': stamp,
            'version': 3,
        }
        if self.display_name:
            metadata['display_name'] = self.display_name
        with open(f"{base}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        self.dirty = False
        self.update_caption()
        self.show_toast("Pista guardada")
        print(f"✓ Pista guardada: {name}")

    def load_track(self, name):
        base = os.path.join('tracks', name)
        with open(f"{base}.json", 'r') as f:
            meta = json.load(f)

        def load_layer(path, target, keyed):
            if not os.path.exists(path):
                return
            img = pygame.image.load(path)
            if img.get_size() != (TRACK_W, TRACK_H):
                img = pygame.transform.scale(img, (TRACK_W, TRACK_H))
            target.fill(BLACK)
            if keyed:
                img.set_colorkey(BLACK)
            target.blit(img, (0, 0))

        load_layer(f"{base}_track.png", self.road, False)
        load_layer(f"{base}_speed.png", self.speed, True)
        load_layer(f"{base}_slow.png", self.slow, True)

        sx = TRACK_W / meta.get('width', TRACK_W)
        sy = TRACK_H / meta.get('height', TRACK_H)
        sp = meta.get('spawn_point')
        self.spawn = (sp[0] * sx, sp[1] * sy, sp[2] if len(sp) > 2 else -math.pi / 2) if sp else None
        fl = meta.get('finish_line')
        self.finish = (fl[0] * sx, fl[1] * sy, fl[2] * sx, fl[3] * sy) if fl else None
        self.checkpoints = [(c[0] * sx, c[1] * sy, c[2] * sx, c[3] * sy) for c in meta.get('checkpoints', [])]
        self.required_laps = int(meta.get('required_laps', 3))
        self.track_name = name
        self.display_name = meta.get('display_name')
        self.created = meta.get('created')
        self.view_dirty = True
        self.check_dirty = True
        print(f"✓ Pista cargada en el editor: {name}")

    # ------------------------------------------------------------------ #
    # Teclado
    # ------------------------------------------------------------------ #
    def on_key(self, event):
        key, mod = event.key, event.mod
        ctrl = mod & pygame.KMOD_CTRL

        if self.modal == 'save_name':
            if key == pygame.K_RETURN:
                self.display_name = self.name_text.strip() or None
                self.modal = None
                self.save_track()
                if self.exit_after_save:
                    self.running = False
            elif key == pygame.K_ESCAPE:
                self.modal = None
                self.exit_after_save = False
            elif key == pygame.K_BACKSPACE:
                self.name_text = self.name_text[:-1]
            elif event.unicode and event.unicode.isprintable() and len(self.name_text) < 30:
                self.name_text += event.unicode
            return
        if self.modal == 'templates':
            choice = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}.get(key)
            if choice is not None:
                self.modal = None
                self.apply_template(TEMPLATES[choice][0])
            elif key == pygame.K_ESCAPE:
                self.modal = None
            return
        if self.test:
            if key == pygame.K_ESCAPE:
                self.test = None
            elif key == pygame.K_r:
                self.test.restart()
            return
        if self.tool == 'curve' and self.curve_points:
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.finish_curve(closed=False)
                return
            if key == pygame.K_BACKSPACE:
                self.curve_points.pop()
                return
            if key == pygame.K_ESCAPE:
                self.curve_points = []
                return
        if self.modal == 'unsaved':
            if key in (pygame.K_g, pygame.K_RETURN):
                self.modal = None
                self.exit_after_save = True
                self.request_save()
                if not self.modal:
                    self.running = False
            elif key == pygame.K_d:
                self.running = False
            elif key == pygame.K_ESCAPE:
                self.modal = None
            return

        if ctrl and key == pygame.K_z:
            self.redo() if mod & pygame.KMOD_SHIFT else self.undo()
        elif ctrl and key == pygame.K_y:
            self.redo()
        elif ctrl and key == pygame.K_s:
            self.request_save()
        elif key == pygame.K_ESCAPE:
            if self.drag:
                self.drag = None
            elif self.selected:
                self.selected = None
            else:
                self.request_exit()
        elif key in TOOL_KEYS:
            self.do_action('tool:' + TOOL_KEYS[key])
        elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            self.delete_selected()
        elif key == pygame.K_i:
            self.invert_finish()
        elif key == pygame.K_p:
            self.start_test()
        elif key == pygame.K_a:
            self.auto_checkpoints()
        elif key == pygame.K_t:
            self.modal = 'templates'
        elif key == pygame.K_m:
            self.do_action('symmetry')
        elif key in (pygame.K_0, pygame.K_KP0):
            self.reset_zoom()
        elif key == pygame.K_g:
            self.show_grid = not self.show_grid
            self.view_dirty = True
            self.compose_dirty = True
        elif key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.do_action('brush-')
        elif key in (pygame.K_RIGHTBRACKET, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.do_action('brush+')

    # ------------------------------------------------------------------ #
    # Dibujo
    # ------------------------------------------------------------------ #
    def text(self, txt, font, color, **pos):
        surf = font.render(txt, True, color)
        rect = surf.get_rect(**pos)
        self.screen.blit(surf, rect)
        return rect

    def draw_button(self, label, rect, active=False, color=PANEL_LIGHT, swatch=None):
        hovered = rect.collidepoint(pygame.mouse.get_pos()) and not self.modal
        base = tuple(min(255, c + 25) for c in color) if hovered else color
        pygame.draw.rect(self.screen, base, rect, border_radius=6)
        if active:
            pygame.draw.rect(self.screen, YELLOW, rect, 3, border_radius=6)
        x = rect.x + 10
        if swatch:
            pygame.draw.rect(self.screen, swatch, (x, rect.centery - 7, 14, 14), border_radius=3)
            x += 22
            self.text(label, self.small_font, WHITE, midleft=(x, rect.centery))
        else:
            self.text(label, self.small_font, WHITE, center=rect.center)

    def draw_canvas(self):
        if self.view_dirty or self.view_cache is None:
            if self.composed is None or self.compose_dirty:
                composed = self.road.copy()
                composed.blit(self.speed, (0, 0))
                composed.blit(self.slow, (0, 0))
                if self.show_grid:
                    for gx in range(0, TRACK_W, 50):
                        pygame.draw.line(composed, (50, 50, 62), (gx, 0), (gx, TRACK_H), 2)
                    for gy in range(0, TRACK_H, 50):
                        pygame.draw.line(composed, (50, 50, 62), (0, gy), (TRACK_W, gy), 2)
                self.composed = composed
                self.compose_dirty = False
            vis = pygame.Rect(int(self.offset[0]), int(self.offset[1]),
                              int(math.ceil(self.view.w / self.scale)), int(math.ceil(self.view.h / self.scale)))
            vis = vis.clip(self.composed.get_rect())
            self.view_cache = pygame.transform.scale(self.composed.subsurface(vis), self.view.size)
            self.view_dirty = False
        self.screen.blit(self.view_cache, self.view)
        pygame.draw.rect(self.screen, PANEL_LIGHT, self.view.inflate(4, 4), 2)
        if self.zoom > 1:
            self.text(f"Zoom x{self.zoom:.1f}  (0: volver)", self.tiny_font, YELLOW,
                      bottomleft=(self.view.x + 8, self.view.bottom - 6))

    def draw_symmetry_axes(self):
        if not self.symmetry:
            return
        if self.symmetry in (1, 3):
            x = self.to_screen((TRACK_W / 2, 0))[0]
            for y in range(self.view.top, self.view.bottom, 16):
                pygame.draw.line(self.screen, ORANGE, (x, y), (x, y + 8), 2)
        if self.symmetry in (2, 3):
            y = self.to_screen((0, TRACK_H / 2))[1]
            for x in range(self.view.left, self.view.right, 16):
                pygame.draw.line(self.screen, ORANGE, (x, y), (x + 8, y), 2)

    def draw_curve_preview(self):
        if self.tool != 'curve' or not self.curve_points:
            return
        pts = list(self.curve_points)
        mouse = pygame.mouse.get_pos()
        if self.view.collidepoint(mouse):
            pts.append(self.to_canvas(mouse))
        line = [self.to_screen(p) for p in geo.catmull_rom(pts)]
        if len(line) > 1:
            pygame.draw.lines(self.screen, (120, 120, 160), False, line, max(2, int(self.brush * 2 * self.scale)))
            pygame.draw.lines(self.screen, WHITE, False, line, 2)
        for k, p in enumerate(self.curve_points):
            sp = self.to_screen(p)
            pygame.draw.circle(self.screen, GREEN if k == 0 else YELLOW, (int(sp[0]), int(sp[1])), 7 if k == 0 else 5)

    def draw_arrow(self, start, direction, length, color, width=3):
        end = (start[0] + direction[0] * length, start[1] + direction[1] * length)
        pygame.draw.line(self.screen, color, start, end, width)
        ang = math.atan2(direction[1], direction[0])
        for side in (-1, 1):
            a = ang + math.pi + side * 0.5
            pygame.draw.line(self.screen, color, end, (end[0] + math.cos(a) * 10, end[1] + math.sin(a) * 10), width)

    def draw_line_object(self, line, color, width, label=None, selected=False, is_finish=False):
        a, b = self.to_screen(line[:2]), self.to_screen(line[2:4])
        if selected:
            pygame.draw.line(self.screen, YELLOW, a, b, width + 6)
        pygame.draw.line(self.screen, color, a, b, width)
        for p in (a, b):
            pygame.draw.circle(self.screen, WHITE if selected else color, (int(p[0]), int(p[1])), 6 if selected else 4)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if is_finish:
            self.draw_arrow(mid, finish_direction(line), 34, GREEN, 4)
        if label:
            r = self.text(label, self.small_font, WHITE, center=(mid[0], mid[1]))
            pygame.draw.rect(self.screen, color, r.inflate(10, 6), border_radius=8)
            self.text(label, self.small_font, WHITE, center=r.center)

    def draw_objects(self):
        sel = self.selected
        for i, cp in enumerate(self.checkpoints):
            self.draw_line_object(cp, CHECKPOINT_COLOR, 4, str(i + 1), sel == ('checkpoint', i))
        if self.finish:
            self.draw_line_object(self.finish, FINISH_LINE_COLOR, 5, None, sel == ('finish',), is_finish=True)
        if self.spawn:
            p = self.to_screen(self.spawn[:2])
            if sel == ('spawn',):
                pygame.draw.circle(self.screen, YELLOW, (int(p[0]), int(p[1])), 15, 3)
            # Auto como rectángulo orientado
            ang = self.spawn[2]
            car = pygame.Surface((26, 14), pygame.SRCALPHA)
            car.fill(CYAN)
            pygame.draw.rect(car, BLACK, (18, 2, 6, 10))
            rotated = pygame.transform.rotate(car, -math.degrees(ang))
            self.screen.blit(rotated, rotated.get_rect(center=p))
            self.draw_arrow(p, (math.cos(ang), math.sin(ang)), 36, CYAN, 3)

        # Línea en construcción
        if self.drag and self.drag[0] == 'line':
            _, tool, a, b = self.drag
            line = (a[0], a[1], b[0], b[1])
            color = FINISH_LINE_COLOR if tool == 'finish' else CHECKPOINT_COLOR
            self.draw_line_object(line, color, 4, None, False, is_finish=(tool == 'finish'))

    def draw_cursor(self):
        mouse = pygame.mouse.get_pos()
        if self.modal or not self.view.collidepoint(mouse):
            return
        if self.tool in ('road', 'erase', 'speed', 'slow'):
            color = {'road': WHITE, 'erase': RED, 'speed': SPEED_ZONE_COLOR, 'slow': SLOW_ZONE_COLOR}[self.tool]
            pygame.draw.circle(self.screen, color, mouse, max(2, int(self.brush * self.scale)), 2)
        elif self.tool == 'select':
            key, part = self.hit_test(self.to_canvas(mouse))
            if key and key != self.selected:
                obj = self.get_object(key)
                if key[0] == 'spawn':
                    p = self.to_screen(obj[:2])
                    pygame.draw.circle(self.screen, ORANGE, (int(p[0]), int(p[1])), 15, 2)
                else:
                    pygame.draw.line(self.screen, ORANGE, self.to_screen(obj[:2]), self.to_screen(obj[2:4]), 2)

    def draw_toolbar(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, TOOLBAR_W, self.height))
        self.text("EDITOR DE PISTAS", self.title_font, YELLOW, topleft=(12, 18))
        swatches = {t[0]: t[3] for t in TOOLS}
        for bid, label, rect, active in self.toolbar_items():
            if bid.startswith('tool:'):
                tid = bid[5:]
                self.draw_button(label, rect, active, swatch=swatches[tid])
            elif bid == 'test':
                self.draw_button(label, rect, color=(120, 60, 200))
            elif bid == 'save':
                self.draw_button(label, rect, color=(40, 140, 80) if self.dirty else PANEL_LIGHT)
            else:
                self.draw_button(label, rect)

        self.text(f"Pincel: {self.brush}", self.small_font, WHITE, center=(TOOLBAR_W // 2, self.brush_y + 29))
        self.text("Tamaño", self.tiny_font, TEXT_DIM, center=(TOOLBAR_W // 2, self.brush_y + 7))
        self.text(f"Vueltas: {self.required_laps}", self.small_font, WHITE, center=(TOOLBAR_W // 2, self.laps_y + 29))
        self.text("Para ganar", self.tiny_font, TEXT_DIM, center=(TOOLBAR_W // 2, self.laps_y + 7))

    def draw_checklist(self):
        """Recuadro en la esquina del lienzo con lo que falta para entrenar"""
        items = self.checklist()
        box = pygame.Rect(0, 0, 200, 30 + len(items) * 20)
        box.topright = (self.view.right - 8, self.view.top + 8)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((20, 22, 30, 210))
        self.screen.blit(panel, box)
        all_ok = all(ok for _, ok in items)
        self.text("Lista para entrenar" if all_ok else "Falta para entrenar:", self.tiny_font,
                  GREEN if all_ok else ORANGE, topleft=(box.x + 10, box.y + 8))
        for k, (label, ok) in enumerate(items):
            cy = box.y + 28 + k * 20
            pygame.draw.circle(self.screen, GREEN if ok else RED, (box.x + 16, cy + 6), 5)
            self.text(label, self.tiny_font, WHITE if ok else TEXT_DIM, topleft=(box.x + 28, cy))

    def draw_status(self):
        rect = pygame.Rect(TOOLBAR_W, self.height - STATUS_H, self.width - TOOLBAR_W, STATUS_H)
        pygame.draw.rect(self.screen, PANEL, rect)
        if self.test:
            title, help_text = "Probar:", "Flechas: manejar (abajo frena)   R: reiniciar   Esc: volver al editor. Las líneas rojas son los sensores de la IA."
        else:
            tool = {t[0]: t for t in TOOLS}[self.tool]
            title = f"{tool[1]} ({tool[2]}):"
            help_text = TOOL_HELP[self.tool] + "   Ctrl+rueda: zoom. Rueda apretada o Espacio+arrastrar: mover."
        title_rect = self.text(title, self.small_font, YELLOW, midleft=(rect.x + 12, rect.centery))
        max_w = rect.right - title_rect.right - 24
        while help_text and self.tiny_font.size(help_text)[0] > max_w:
            help_text = help_text[:help_text.rfind(' ')] if ' ' in help_text else help_text[:-1]
        self.text(help_text, self.tiny_font, WHITE, midleft=(title_rect.right + 12, rect.centery))

    def modal_box(self):
        box = pygame.Rect(0, 0, 520, 200)
        box.center = (self.width // 2, self.height // 2)
        return box

    @staticmethod
    def template_rect(k, box):
        w = (box.w - 70) // 2
        return pygame.Rect(box.x + 30 + (k % 2) * (w + 10), box.y + 60 + (k // 2) * 46, w, 38)

    def on_modal_click(self, pos):
        if self.modal == 'templates':
            box = self.modal_box()
            for k, (name, _) in enumerate(TEMPLATES):
                if self.template_rect(k, box).collidepoint(pos):
                    self.modal = None
                    self.apply_template(name)
                    return
            if not box.collidepoint(pos):
                self.modal = None

    def draw_modal(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))
        box = self.modal_box()
        pygame.draw.rect(self.screen, PANEL, box, border_radius=14)
        pygame.draw.rect(self.screen, YELLOW, box, 3, border_radius=14)
        if self.modal == 'templates':
            self.text("Elegí una plantilla", self.font, YELLOW, center=(box.centerx, box.top + 32))
            for k, (_, label) in enumerate(TEMPLATES):
                r = self.template_rect(k, box)
                self.draw_button(f"{k + 1}. {label}", r, color=BLUE)
            self.text("Reemplaza la pista actual (Ctrl+Z para volver)   Esc: cancelar", self.tiny_font, TEXT_DIM,
                      center=(box.centerx, box.bottom - 22))
        elif self.modal == 'save_name':
            self.text("Nombre de la pista", self.font, YELLOW, center=(box.centerx, box.top + 32))
            field = pygame.Rect(box.left + 30, box.top + 60, box.w - 60, 44)
            pygame.draw.rect(self.screen, BG, field, border_radius=6)
            pygame.draw.rect(self.screen, WHITE, field, 2, border_radius=6)
            cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            self.text(self.name_text + cursor, self.font, WHITE, midleft=(field.left + 12, field.centery))
            self.text("Enter: guardar    Esc: cancelar", self.small_font, TEXT_DIM, center=(box.centerx, box.bottom - 40))
        else:
            self.text("Hay cambios sin guardar", self.font, YELLOW, center=(box.centerx, box.top + 40))
            self.text("G: guardar y salir     D: descartar y salir     Esc: seguir editando",
                      self.small_font, WHITE, center=(box.centerx, box.top + 110))

    def draw_test_hud(self):
        lines = self.test.hud_lines()
        box = pygame.Rect(self.view.right - 220, self.view.top + 8, 212, 30 + len(lines) * 22)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((20, 22, 30, 220))
        self.screen.blit(panel, box)
        self.text("MODO PRUEBA", self.small_font, YELLOW, topleft=(box.x + 10, box.y + 8))
        for k, line in enumerate(lines):
            self.text(line, self.small_font, WHITE, topleft=(box.x + 10, box.y + 30 + k * 22))
        msg, color = self.test.message
        w = self.font.size(msg)[0] + 40
        mbox = pygame.Rect(self.view.centerx - w // 2, self.view.bottom - 56, w, 40)
        pygame.draw.rect(self.screen, (20, 22, 30), mbox, border_radius=10)
        pygame.draw.rect(self.screen, color, mbox, 2, border_radius=10)
        self.text(msg, self.font, color, center=mbox.center)

    def draw_toast(self):
        if not self.toast:
            return
        text, color, expires = self.toast
        if pygame.time.get_ticks() > expires:
            self.toast = None
            return
        w = self.small_font.size(text)[0] + 40
        box = pygame.Rect(self.view.left + 12, self.view.top + 12, w, 38)
        pygame.draw.rect(self.screen, PANEL_LIGHT, box, border_radius=10)
        pygame.draw.rect(self.screen, color, box, 2, border_radius=10)
        self.text(text, self.small_font, WHITE, center=box.center)

    def draw(self):
        self.screen.fill(BG)
        self.draw_canvas()
        self.screen.set_clip(self.view)
        self.draw_symmetry_axes()
        self.draw_objects()
        if self.test:
            self.test.draw(self.screen, self.to_screen, self.scale, self.font, self.small_font)
            self.draw_test_hud()
        else:
            self.draw_curve_preview()
            self.draw_cursor()
            self.draw_checklist()
        self.screen.set_clip(None)
        self.draw_toolbar()
        self.draw_status()
        self.draw_toast()
        if self.modal:
            self.draw_modal()
        pygame.display.flip()

    # ------------------------------------------------------------------ #
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.test:
                        self.test = None
                    self.request_exit()
                elif event.type == pygame.KEYDOWN:
                    self.on_key(event)
                elif self.modal:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.on_modal_click(event.pos)
                elif self.test:
                    continue
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
                    if event.pos[0] < TOOLBAR_W:
                        if event.button == 1:
                            self.on_toolbar_click(event.pos)
                    else:
                        self.on_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 2, 3):
                    self.on_mouse_up(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.on_mouse_move(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.zoom_at(pygame.mouse.get_pos(), 1.2 if event.y > 0 else 1 / 1.2)
                    else:
                        self.do_action('brush+' if event.y > 0 else 'brush-')
            if self.test:
                self.test.update(pygame.key.get_pressed())
            self.draw()
            self.clock.tick(60)
        pygame.quit()
        print("✓ Editor de pistas cerrado")


# Alias para que el selector pueda usarlo igual que V2
TrackEditor = TrackEditorV3


if __name__ == "__main__":
    TrackEditorV3().run()
