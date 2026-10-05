"""
Menús del entrenamiento:
- main_menu: elegir escenario (pista o laberinto)
- maze_settings: opciones del laberinto con vista previa
- workshop: "taller del agente", donde se configura qué ve, cómo piensa,
  cómo maneja y cómo evoluciona el auto
"""

import math
import os
from dataclasses import replace

import numpy as np
import pygame

from ai import ui
from ai.config import (AgentConfig, ACTION_SETS, BRAIN_SIZES, PRESETS, brain_path, list_profiles)
from ai.scenarios import LaberintoScenario, MAZE_SIZES
from ai.world import World


def _quit_event(event):
    return event.type == pygame.QUIT


def loading(screen, message):
    """Pantalla de espera mientras se prepara algo pesado"""
    screen.fill(ui.BG)
    fonts = ui.Fonts()
    ui.text(screen, message, fonts.big, ui.WHITE, center=screen.get_rect().center)
    pygame.display.flip()
    pygame.event.pump()


# ---------------------------------------------------------------------- #
# Menú principal
# ---------------------------------------------------------------------- #
MAIN_OPTIONS = [
    ('pista', "Recorrer una pista", "Los autos aprenden a dar vueltas a una pista hecha con el editor, "
                                     "pasando los checkpoints en orden.", ui.BLUE),
    ('laberinto', "Resolver un laberinto", "Los autos tienen que encontrar la salida de un laberinto "
                                           "generado al azar. Puede cambiar cada tantas generaciones.", ui.PURPLE),
]


def main_menu(screen):
    """Devuelve 'pista', 'laberinto' o 'quit'"""
    fonts = ui.Fonts()
    clock = pygame.time.Clock()
    selected = 0
    while True:
        W, H = screen.get_size()
        cw, ch = min(420, (W - 120) // 2), 260
        top = H // 2 - ch // 2 + 10
        cards = [pygame.Rect(W // 2 - cw - 20 + i * (cw + 40), top, cw, ch) for i in range(len(MAIN_OPTIONS))]
        quit_rect = pygame.Rect(W // 2 - 90, top + ch + 40, 180, 44)

        for event in pygame.event.get():
            if _quit_event(event):
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    selected = 0
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    selected = 1
                elif event.key in (pygame.K_1, pygame.K_2):
                    return MAIN_OPTIONS[event.key - pygame.K_1][0]
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return MAIN_OPTIONS[selected][0]
            if event.type == pygame.MOUSEMOTION:
                for i, r in enumerate(cards):
                    if r.collidepoint(event.pos):
                        selected = i
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(cards):
                    if r.collidepoint(event.pos):
                        return MAIN_OPTIONS[i][0]
                if quit_rect.collidepoint(event.pos):
                    return 'quit'

        screen.fill(ui.BG)
        ui.text(screen, "AUTOS QUE APRENDEN SOLOS", fonts.title, ui.YELLOW, center=(W // 2, top - 110))
        ui.text(screen, "Algoritmo genético + redes neuronales  ·  ¿Qué tienen que aprender?",
                fonts.normal, ui.TEXT_DIM, center=(W // 2, top - 68))
        for i, ((key, title, desc, color), r) in enumerate(zip(MAIN_OPTIONS, cards)):
            pygame.draw.rect(screen, ui.PANEL_LIGHT if i == selected else ui.PANEL, r, border_radius=14)
            if i == selected:
                pygame.draw.rect(screen, ui.YELLOW, r, 3, border_radius=14)
            icon = pygame.Rect(r.x + 24, r.y + 24, 64, 64)
            pygame.draw.rect(screen, color, icon, border_radius=12)
            _draw_icon(screen, key, icon)
            ui.text(screen, f"{i + 1}", fonts.small, ui.TEXT_DIM, topright=(r.right - 18, r.y + 18))
            ui.text(screen, title, fonts.big, ui.WHITE, topleft=(r.x + 24, r.y + 110))
            _wrap(screen, desc, fonts.small, ui.TEXT_DIM, pygame.Rect(r.x + 24, r.y + 150, r.w - 48, 100))
        ui.button(screen, fonts.normal, "Salir (Esc)", quit_rect)
        ui.text(screen, "Flechas + Enter, o clic", fonts.tiny, ui.TEXT_DIM, center=(W // 2, H - 24))
        pygame.display.flip()
        clock.tick(60)


def _draw_icon(surf, key, r):
    if key == 'pista':
        pygame.draw.ellipse(surf, ui.WHITE, r.inflate(-14, -26), 6)
        pygame.draw.rect(surf, ui.RED, (r.centerx - 3, r.bottom - 24, 6, 14))
    else:
        c = (255, 255, 255)
        x, y, s = r.x + 12, r.y + 12, 10
        for (a, b, w, h) in ((0, 0, 4, 1), (0, 0, 1, 3), (2, 1, 1, 3), (0, 3, 3, 1), (4, 1, 1, 4), (1, 2, 2, 1)):
            pygame.draw.rect(surf, c, (x + a * s, y + b * s, max(3, w * s), max(3, h * s)))


def _wrap(surf, txt, font, color, rect, line_h=22):
    words, line, y = txt.split(), "", rect.y
    for w in words:
        test = (line + " " + w).strip()
        if font.size(test)[0] > rect.w and line:
            ui.text(surf, line, font, color, topleft=(rect.x, y))
            y += line_h
            line = w
        else:
            line = test
    if line:
        ui.text(surf, line, font, color, topleft=(rect.x, y))
    return y + line_h


# ---------------------------------------------------------------------- #
# Opciones del laberinto
# ---------------------------------------------------------------------- #
MAZE_STEPPERS = [
    ui.Stepper('size', "Tamaño", options=[(k, k) for k in MAZE_SIZES],
               help_text="Chico 9x6, mediano 13x9, grande 18x12 celdas. Conviene empezar con el chico."),
    ui.Stepper('new_every', "Laberinto nuevo cada", options=[(0, "nunca"), (1, "1 gen."), (3, "3 gen."),
                                                             (5, "5 gen."), (10, "10 gen."), (25, "25 gen.")],
               help_text="Si cambia, los autos no pueden memorizar el camino: aprenden a resolver cualquier laberinto."),
    ui.Stepper('braid', "Con atajos (varios caminos)",
               help_text="Agrega pasajes extra: hay más de un camino y menos callejones sin salida."),
]


def maze_settings(screen, opts):
    """opts: dict con size, new_every, braid (se modifica). Devuelve LaberintoScenario o None."""
    fonts = ui.Fonts()
    clock = pygame.time.Clock()
    scenario = LaberintoScenario(opts['size'], opts['new_every'], opts['braid'])
    preview = None
    while True:
        W, H = screen.get_size()
        left = pygame.Rect(40, 110, 430, H - 220)
        for i, st in enumerate(MAZE_STEPPERS):
            st.layout(left.x, left.y + i * 48, left.w, 38)
        regen = pygame.Rect(left.x, left.y + len(MAZE_STEPPERS) * 48 + 20, left.w, 42)
        back = pygame.Rect(40, H - 80, 200, 50)
        go = pygame.Rect(W - 300, H - 80, 260, 50)
        prev_area = pygame.Rect(left.right + 40, 110, W - left.right - 80, H - 220)
        if preview is None:
            mw, mh = scenario.size
            s = min(prev_area.w / mw, prev_area.h / mh)
            preview = pygame.transform.smoothscale(scenario.background, (int(mw * s), int(mh * s)))
            sx, sy, sa = scenario.spawn
            preview_spawn = (sx * s, sy * s, sa)

        rebuild = False
        for event in pygame.event.get():
            if _quit_event(event):
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN:
                    return scenario
                if event.key == pygame.K_g:
                    rebuild = True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for st in MAZE_STEPPERS:
                    new = st.click(event.pos, opts[st.key])
                    if new is not None and new != opts[st.key]:
                        opts[st.key] = new
                        rebuild = st.key != 'new_every'
                        scenario.new_every = opts['new_every']
                if regen.collidepoint(event.pos):
                    rebuild = True
                if back.collidepoint(event.pos):
                    return None
                if go.collidepoint(event.pos):
                    return scenario
        if rebuild:
            scenario = LaberintoScenario(opts['size'], opts['new_every'], opts['braid'])
            preview = None
            continue

        screen.fill(ui.BG)
        ui.text(screen, "LABERINTO", fonts.title, ui.YELLOW, topleft=(40, 30))
        ui.text(screen, "Los autos salen de la esquina y tienen que llegar a la bandera a cuadros",
                fonts.small, ui.TEXT_DIM, topleft=(42, 74))
        mouse = pygame.mouse.get_pos()
        hover_help = ""
        for st in MAZE_STEPPERS:
            hovered = st.rect.collidepoint(mouse)
            st.draw(screen, fonts, opts[st.key], highlight=hovered)
            if hovered:
                hover_help = st.help
        ui.button(screen, fonts.normal, "Generar otro (G)", regen)
        y = regen.bottom + 24
        for st in MAZE_STEPPERS:
            y = _wrap(screen, f"{st.label}: {st.help}", fonts.tiny,
                      ui.WHITE if st.help == hover_help else ui.TEXT_DIM, pygame.Rect(left.x, y, left.w, 80), 18) + 6

        r = preview.get_rect(center=prev_area.center)
        screen.blit(preview, r)
        px, py, pa = preview_spawn
        car = (r.x + px, r.y + py)
        tip = (car[0] + math.cos(pa) * 22, car[1] + math.sin(pa) * 22)
        pygame.draw.circle(screen, ui.RED, (int(car[0]), int(car[1])), 8)
        pygame.draw.line(screen, ui.RED, car, tip, 3)

        ui.button(screen, fonts.normal, "Volver (Esc)", back)
        ui.button(screen, fonts.normal, "Siguiente (Enter)", go, ui.GREEN)
        pygame.display.flip()
        clock.tick(60)


# ---------------------------------------------------------------------- #
# Taller del agente
# ---------------------------------------------------------------------- #
ACTION_SHORT = {'simple': "Simple", 'clasico': "Clásico", 'completo': "Completo"}

SECTIONS = [
    ("QUÉ VE", [
        ui.Stepper('num_sensors', "Sensores", lo=1, hi=15, step=1,
                   help_text="Rayos que miden la distancia a la pared. Más sensores = ve mejor, pero tarda más en aprender."),
        ui.Stepper('sensor_spread', "Ángulo de visión", lo=30, hi=360, step=15, fmt="{}°",
                   help_text="Qué tan abiertos están los sensores. 360° = ve para todos lados."),
        ui.Stepper('sensor_range', "Alcance", lo=60, hi=500, step=20, fmt="{} px",
                   help_text="Hasta qué distancia llegan los sensores."),
        ui.Stepper('use_speed', "Siente su velocidad",
                   help_text="Le pasa su propia velocidad a la red. Ayuda a frenar antes de las curvas."),
        ui.Stepper('use_compass', "Brújula al objetivo",
                   help_text="Le dice hacia dónde queda el próximo checkpoint o la salida (en línea recta)."),
    ]),
    ("CÓMO PIENSA", [
        ui.Stepper('brain_size', "Tamaño de la red",
                   options=[(k, k) for k in BRAIN_SIZES],
                   help_text="Neuronas en las capas ocultas: " + ", ".join(f"{k} {'-'.join(map(str, v))}" for k, v in BRAIN_SIZES.items()) + ". Más grande puede aprender más, pero más lento."),
    ]),
    ("CÓMO MANEJA", [
        ui.Stepper('action_set', "Acciones", options=[(k, ACTION_SHORT[k]) for k in ACTION_SETS],
                   help_text=" | ".join(v['label'] for v in ACTION_SETS.values())),
        ui.Stepper('max_speed', "Vel. máxima", lo=2.0, hi=12.0, step=0.5, fmt="{:.1f}",
                   help_text="Más rápido = vueltas más cortas, pero más difícil no chocar."),
        ui.Stepper('turn_speed', "Giro", lo=0.03, hi=0.25, step=0.01, fmt="{:.2f}",
                   help_text="Cuánto dobla por paso (en radianes). Los laberintos necesitan girar más."),
    ]),
    ("CÓMO EVOLUCIONA", [
        ui.Stepper('population', "Autos por gen.", lo=10, hi=300, step=10,
                   help_text="Más autos = más variedad para elegir a los mejores, pero cada generación tarda más."),
        ui.Stepper('elite_pct', "Élite", lo=5, hi=50, step=5, fmt="{} %",
                   help_text="Porcentaje de los mejores que pasan sin cambios a la siguiente generación."),
        ui.Stepper('mutation_rate', "Mutación: prob.", lo=0.02, hi=0.6, step=0.02, fmt="{:.2f}",
                   help_text="Probabilidad de que cada peso de la red cambie al azar en los hijos."),
        ui.Stepper('mutation_strength', "Mutación: fuerza", lo=0.05, hi=1.5, step=0.05, fmt="{:.2f}",
                   help_text="Qué tan grande es cada cambio. Grande = explora más; chico = afina."),
        ui.Stepper('crossover', "Cruza entre padres",
                   help_text="Los hijos mezclan neuronas de dos padres en vez de copiar a uno solo."),
        ui.Stepper('max_steps', "Pasos por gen.", lo=500, hi=10000, step=250,
                   help_text="Tiempo máximo de cada generación."),
        ui.Stepper('patience', "Paciencia", lo=50, hi=1500, step=50,
                   help_text="Pasos sin avanzar permitidos. Si un auto no avanza en tantos pasos, queda afuera (evita autos dando vueltas en el lugar)."),
    ]),
]
ALL_STEPPERS = [st for _, items in SECTIONS for st in items]


def brain_status(profile, scenario, cfg):
    """(existe, compatible, texto)"""
    path = brain_path(profile, scenario.key)
    if not os.path.exists(path):
        return False, False, "Todavía no hay un cerebro entrenado con este perfil para este escenario."
    try:
        data = np.load(path)
        sizes = list(data['sizes'])
        gen = int(data['generation']) if 'generation' in data else None
    except Exception:
        return True, False, "El cerebro guardado está dañado."
    if sizes != cfg.layer_sizes():
        return True, False, ("Hay un cerebro guardado, pero con otra forma de red "
                             f"({'-'.join(map(str, sizes))}). Cambiaste sensores, red o acciones.")
    extra = f" (generación {gen})" if gen else ""
    return True, True, f"Hay un cerebro entrenado{extra}: podés seguir entrenándolo."


class Workshop:
    def __init__(self, screen, scenario, cfg, profile):
        self.screen = screen
        self.fonts = ui.Fonts()
        self.scenario = scenario
        self.cfg = replace(cfg)
        self.name = ui.TextInput(profile)
        self.editing_name = False
        self.toast = ("", 0)
        self.profile_page = 0
        self._bg_cache = None

    # -------------------------------------------------------------- #
    def layout(self):
        W, H = self.screen.get_size()
        col_w = min(360, (W - 520) // 2)
        self.cols = [pygame.Rect(30, 120, col_w, 0), pygame.Rect(30 + col_w + 20, 120, col_w, 0)]
        row_h, head_h = 31, 26
        # Columna 1: ve / piensa / maneja. Columna 2: evoluciona
        placement = [(0, SECTIONS[0]), (0, SECTIONS[1]), (0, SECTIONS[2]), (1, SECTIONS[3])]
        ys = [self.cols[0].y, self.cols[1].y]
        self.headers = []
        for col, (title, items) in placement:
            x = self.cols[col].x
            self.headers.append((title, (x, ys[col])))
            ys[col] += head_h
            for st in items:
                st.layout(x, ys[col], col_w, row_h - 3)
                ys[col] += row_h
            ys[col] += 8
        self.right = pygame.Rect(self.cols[1].right + 25, 70, W - self.cols[1].right - 50, H - 160)

        # Presets y perfil, debajo de la columna 2
        y = ys[1] + 4
        self.presets_y = y
        pw = (col_w - 6) // 2
        self.preset_rects = [(name, pygame.Rect(self.cols[1].x + (i % 2) * (pw + 6), y + 24 + (i // 2) * 34, pw, 28))
                             for i, name in enumerate(PRESETS)]
        y += 24 + ((len(PRESETS) + 1) // 2) * 34 + 10
        self.profile_y = y
        self.name_rect = pygame.Rect(self.cols[1].x, y + 24, col_w - 100, 30)
        self.save_rect = pygame.Rect(self.name_rect.right + 6, y + 24, 94, 30)
        y += 60
        profiles = [p for p in list_profiles() if not p.startswith('_')]
        self.profile_rects = []
        x = self.cols[1].x
        for p in profiles:
            w = min(col_w, self.fonts.tiny.size(p)[0] + 18)
            if x + w > self.cols[1].right:
                x = self.cols[1].x
                y += 28
            if y > H - 110:
                break
            self.profile_rects.append((p, pygame.Rect(x, y, w, 24)))
            x += w + 6

        bw = 230
        self.btn_back = pygame.Rect(30, H - 74, 180, 48)
        self.btn_new = pygame.Rect(W - 2 * bw - 40, H - 74, bw, 48)
        self.btn_continue = pygame.Rect(W - bw - 30, H - 74, bw, 48)

    # -------------------------------------------------------------- #
    def flash(self, msg):
        self.toast = (msg, pygame.time.get_ticks() + 2500)

    def handle_click(self, pos):
        self.editing_name = self.name_rect.collidepoint(pos)
        for st in ALL_STEPPERS:
            value = getattr(self.cfg, st.key)
            new = st.click(pos, value)
            if new is not None:
                setattr(self.cfg, st.key, new)
                return None
        for name, r in self.preset_rects:
            if r.collidepoint(pos):
                self.cfg = replace(PRESETS[name])
                self.flash(f"Preset \"{name}\" aplicado")
                return None
        if self.save_rect.collidepoint(pos):
            self.save_profile()
        for p, r in self.profile_rects:
            if r.collidepoint(pos):
                try:
                    self.cfg = AgentConfig.load(p)
                    self.name.value = p
                    self.flash(f"Perfil \"{p}\" cargado")
                except Exception:
                    self.flash("No se pudo cargar ese perfil")
        if self.btn_back.collidepoint(pos):
            return 'back'
        if self.btn_new.collidepoint(pos):
            return 'new'
        if self.btn_continue.collidepoint(pos) and brain_status(self.profile, self.scenario, self.cfg)[1]:
            return 'continue'
        return None

    @property
    def profile(self):
        name = "".join(c for c in self.name.value.strip() if c.isalnum() or c in "-_ ").strip()
        return name or "perfil"

    def save_profile(self):
        self.cfg.save(self.profile)
        self.flash(f"Perfil \"{self.profile}\" guardado")

    # -------------------------------------------------------------- #
    def draw_preview(self, rect):
        """Lo que 've' el auto en la salida: recorte del mapa con los sensores"""
        sc = self.scenario
        pygame.draw.rect(self.screen, (28, 30, 40), rect, border_radius=8)
        world = World(sc.road_mask, self.cfg)
        sx, sy, sa = sc.spawn
        world.reset(1, sx, sy, sa)
        world.read_sensors(np.array([0]))

        if self._bg_cache is None:
            self._bg_cache = sc.background
        reach = self.cfg.sensor_range + 40
        src = pygame.Rect(int(sx - reach), int(sy - reach), 2 * reach, 2 * reach)
        crop = pygame.Surface(src.size)
        crop.fill(ui.BG)
        crop.blit(self._bg_cache, (0, 0), src)
        side = min(rect.w, rect.h) - 16
        view = pygame.transform.smoothscale(crop, (side, side))
        vr = view.get_rect(center=rect.center)
        self.screen.blit(view, vr)
        s = side / src.w

        def tp(x, y):
            return (vr.x + (x - src.x) * s, vr.y + (y - src.y) * s)
        car = tp(sx, sy)
        for ang, dist in zip(world.sensor_angles, world.sensors[0]):
            a = sa + float(ang)
            end = tp(sx + math.cos(a) * float(dist), sy + math.sin(a) * float(dist))
            full = tp(sx + math.cos(a) * self.cfg.sensor_range, sy + math.sin(a) * self.cfg.sensor_range)
            pygame.draw.line(self.screen, (120, 60, 60), car, full, 1)
            pygame.draw.line(self.screen, (255, 90, 90), car, end, 2)
            pygame.draw.circle(self.screen, (255, 60, 60), (int(end[0]), int(end[1])), 4)
        L, Wd = 20 * s, 10 * s
        ca, sn = math.cos(sa), math.sin(sa)
        pts = [(car[0] + ca * dx - sn * dy, car[1] + sn * dx + ca * dy)
               for dx, dy in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd))]
        pygame.draw.polygon(self.screen, ui.YELLOW, pts)

    def draw(self):
        f, scr = self.fonts, self.screen
        W, H = scr.get_size()
        self.layout()
        scr.fill(ui.BG)
        ui.text(scr, "TALLER DEL AGENTE", f.title, ui.YELLOW, topleft=(30, 24))
        ui.text(scr, f"Escenario: {self.scenario.title}  ·  {getattr(self.scenario, 'name', '') or self.scenario.goal_text}",
                f.small, ui.TEXT_DIM, topleft=(32, 66))
        ui.text(scr, "Pasá el mouse sobre una opción para ver qué hace", f.tiny, ui.TEXT_DIM, topleft=(32, 88))

        mouse = pygame.mouse.get_pos()
        hover = None
        for title, pos in self.headers:
            ui.text(scr, title, f.small, ui.CYAN, topleft=(pos[0] + 2, pos[1] + 4))
        for st in ALL_STEPPERS:
            h = st.rect.collidepoint(mouse)
            st.draw(scr, f, getattr(self.cfg, st.key), highlight=h)
            if h:
                hover = st

        ui.text(scr, "PRESETS", f.small, ui.CYAN, topleft=(self.cols[1].x + 2, self.presets_y + 4))
        for name, r in self.preset_rects:
            ui.button(scr, f.tiny, name, r)
        ui.text(scr, "PERFIL", f.small, ui.CYAN, topleft=(self.cols[1].x + 2, self.profile_y + 4))
        pygame.draw.rect(scr, (28, 30, 40), self.name_rect, border_radius=6)
        pygame.draw.rect(scr, ui.YELLOW if self.editing_name else ui.PANEL_LIGHT, self.name_rect, 2, border_radius=6)
        cursor = "|" if self.editing_name and (pygame.time.get_ticks() // 500) % 2 else ""
        ui.text(scr, ui.fit(self.name.value + cursor, f.small, self.name_rect.w - 12), f.small, ui.WHITE,
                midleft=(self.name_rect.x + 8, self.name_rect.centery))
        ui.button(scr, f.small, "Guardar", self.save_rect)
        for p, r in self.profile_rects:
            ui.button(scr, f.tiny, p, r, active=(p == self.profile))

        # Derecha: lo que ve + la red
        right = self.right
        prev_h = min(right.w, int(right.h * 0.52))
        ui.text(scr, "Lo que ve en la salida", f.small, ui.WHITE, topleft=(right.x, right.y))
        self.draw_preview(pygame.Rect(right.x, right.y + 24, right.w, prev_h))
        y = right.y + 24 + prev_h + 12
        sizes = self.cfg.layer_sizes()
        ui.text(scr, f"Su red: {' - '.join(map(str, sizes))}   ({self.cfg.num_params()} pesos)",
                f.small, ui.WHITE, topleft=(right.x, y))
        y += 24
        inputs = [f"{self.cfg.num_sensors} sensores"]
        if self.cfg.use_speed:
            inputs.append("velocidad")
        if self.cfg.use_compass:
            inputs.append("brújula (2)")
        ui.text(scr, ui.fit("Entradas: " + ", ".join(inputs), f.tiny, right.w), f.tiny, ui.TEXT_DIM, topleft=(right.x, y))
        y += 20
        net_h = right.bottom - y
        if net_h > 60:
            ui.draw_network(scr, f, pygame.Rect(right.x, y, right.w, net_h), sizes)

        # Ayuda contextual
        help_rect = pygame.Rect(self.btn_back.right + 20, H - 78, self.btn_new.x - self.btn_back.right - 40, 56)
        exists, ok, status = brain_status(self.profile, self.scenario, self.cfg)
        if hover:
            _wrap(scr, hover.help, f.small, ui.WHITE, help_rect, 20)
        else:
            _wrap(scr, status, f.small, ui.GREEN if ok else ui.TEXT_DIM, help_rect, 20)

        ui.button(scr, f.normal, "Volver (Esc)", self.btn_back)
        ui.button(scr, f.normal, "Empezar de cero", self.btn_new, ui.BLUE)
        ui.button(scr, f.normal, "Seguir entrenando", self.btn_continue, ui.GREEN, enabled=ok)

        msg, until = self.toast
        if msg and pygame.time.get_ticks() < until:
            r = ui.text(scr, msg, f.small, ui.WHITE, midtop=(W // 2, 14))
            pygame.draw.rect(scr, ui.PANEL_LIGHT, r.inflate(24, 12), border_radius=8)
            ui.text(scr, msg, f.small, ui.WHITE, midtop=(W // 2, 14))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if _quit_event(event):
                    return 'quit'
                if event.type == pygame.KEYDOWN:
                    if self.editing_name:
                        if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                            self.editing_name = False
                        else:
                            self.name.key(event)
                        continue
                    if event.key == pygame.K_ESCAPE:
                        return 'back'
                    if event.key == pygame.K_RETURN:
                        return 'new'
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    result = self.handle_click(event.pos)
                    if result:
                        return result
            self.draw()
            clock.tick(60)


def workshop(screen, scenario, cfg, profile):
    """Devuelve (acción, config, perfil). acción: 'new' | 'continue' | 'back' | 'quit'"""
    ws = Workshop(screen, scenario, cfg, profile)
    action = ws.run()
    if action in ('new', 'continue'):
        ws.cfg.save(ws.profile)
    return action, ws.cfg, ws.profile
