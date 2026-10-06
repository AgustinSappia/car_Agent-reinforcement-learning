"""
Menús del entrenamiento:
- main_menu: elegir escenario (pista o laberinto)
- maze_settings: opciones del laberinto con vista previa
- workshop: "taller del agente", donde se configura qué ve, cómo piensa,
  cómo maneja, cómo evoluciona y qué premia su puntaje (distinto para cada tipo de agente)
"""

import json
import math
import os
import numpy as np
import pygame

from ai import ui
from ai.brain import read_brain
from ai.config import (AgentConfig, ACTION_SETS, BRAIN_SIZES, PRESETS, KIND_LABELS, brain_path, list_profiles,
                       preset, copy_config, random_color, used_colors)
from ai.detail import agent_detail
from ai.sprites import draw_car
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
TRAIN_OPTIONS = [
    ('pista', "Recorrer una pista", "Dar vueltas a una pista del editor pasando los checkpoints.", ui.BLUE),
    ('varias', "Varias pistas", "El mismo cerebro aprende una pista tras otra: maneja en cualquiera.", (40, 110, 200)),
    ('laberinto', "Resolver un laberinto", "Encontrar la salida de un laberinto generado al azar.", ui.PURPLE),
    ('futbol', "Jugar al fútbol", "Llevar la pelota al arco rival, contra un bot o entre ellos.", (40, 150, 70)),
]
SHOW_OPTIONS = [
    ('campeones', "Campeones", "Los cerebros entrenados: miralos contra la generación 1 (Modo Expo) o jugá contra ellos.",
     (200, 120, 30)),
]
MAIN_OPTIONS = TRAIN_OPTIONS + SHOW_OPTIONS


def main_menu(screen):
    """Devuelve la clave de la opción elegida o 'quit'"""
    fonts = ui.Fonts()
    clock = pygame.time.Clock()
    selected = 0
    while True:
        W, H = screen.get_size()
        gap = 18
        cw = min(260, (W - 80 - gap * 3) // 4)
        ch = 170
        top1 = 170
        top2 = top1 + ch + 70
        cards = []
        for row, (options, top) in enumerate(((TRAIN_OPTIONS, top1), (SHOW_OPTIONS, top2))):
            total = len(options) * cw + (len(options) - 1) * gap
            x0 = (W - total) // 2
            cards += [pygame.Rect(x0 + i * (cw + gap), top, cw, ch) for i in range(len(options))]
        quit_rect = pygame.Rect(W // 2 - 90, min(H - 64, top2 + ch + 28), 180, 42)

        for event in pygame.event.get():
            if _quit_event(event):
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    selected = (selected - 1) % len(MAIN_OPTIONS)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    selected = (selected + 1) % len(MAIN_OPTIONS)
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    selected = len(TRAIN_OPTIONS) if selected < len(TRAIN_OPTIONS) else 0
                elif pygame.K_1 <= event.key < pygame.K_1 + len(MAIN_OPTIONS):
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
        ui.text(screen, "AUTOS QUE APRENDEN SOLOS", fonts.title, ui.YELLOW, center=(W // 2, 52))
        ui.text(screen, "Algoritmo genético + redes neuronales", fonts.normal, ui.TEXT_DIM, center=(W // 2, 92))
        ui.text(screen, "ENTRENAR", fonts.small, ui.CYAN, midbottom=(W // 2, top1 - 10))
        ui.text(screen, "MOSTRAR EN LA EXPO", fonts.small, ui.CYAN, midbottom=(W // 2, top2 - 10))
        for i, ((key, title, desc, color), r) in enumerate(zip(MAIN_OPTIONS, cards)):
            pygame.draw.rect(screen, ui.PANEL_LIGHT if i == selected else ui.PANEL, r, border_radius=14)
            if i == selected:
                pygame.draw.rect(screen, ui.YELLOW, r, 3, border_radius=14)
            icon = pygame.Rect(r.x + 16, r.y + 16, 48, 48)
            pygame.draw.rect(screen, color, icon, border_radius=10)
            _draw_icon(screen, key, icon)
            ui.text(screen, f"{i + 1}", fonts.small, ui.TEXT_DIM, topright=(r.right - 14, r.y + 14))
            ui.text(screen, ui.fit(title, fonts.normal, r.w - 28), fonts.normal, ui.WHITE, topleft=(r.x + 16, r.y + 76))
            _wrap(screen, desc, fonts.tiny, ui.TEXT_DIM, pygame.Rect(r.x + 16, r.y + 104, r.w - 32, 60), 18)
        ui.button(screen, fonts.normal, "Salir (Esc)", quit_rect)
        pygame.display.flip()
        clock.tick(60)


def _draw_icon(surf, key, r):
    white = (255, 255, 255)
    c = r.center
    if key in ('pista', 'varias'):
        pygame.draw.ellipse(surf, white, r.inflate(-12, -20), 5)
        if key == 'varias':
            pygame.draw.ellipse(surf, white, r.inflate(-26, -32), 3)
    elif key == 'laberinto':
        x, y, s = r.x + 8, r.y + 8, 8
        for (a, b, w, h) in ((0, 0, 4, 1), (0, 0, 1, 3), (2, 1, 1, 3), (0, 3, 3, 1), (4, 1, 1, 4), (1, 2, 2, 1)):
            pygame.draw.rect(surf, white, (x + a * s, y + b * s, max(3, w * s), max(3, h * s)))
    elif key == 'futbol':
        pygame.draw.circle(surf, white, c, 15)
        pygame.draw.circle(surf, (20, 20, 20), c, 15, 2)
        pygame.draw.circle(surf, (20, 20, 20), c, 5)
    elif key == 'expo':
        pygame.draw.rect(surf, white, (r.x + 8, r.y + 12, 14, 24), 2)
        pygame.draw.rect(surf, white, (r.x + 26, r.y + 12, 14, 24))
    elif key == 'competir':
        pygame.draw.polygon(surf, white, [(r.x + 14, r.y + 12), (r.x + 36, c[1]), (r.x + 14, r.bottom - 12)])
    else:
        pygame.draw.polygon(surf, white, [(r.x + 10, r.y + 14), (r.x + 18, r.y + 26), (c[0], r.y + 10),
                                          (r.right - 18, r.y + 26), (r.right - 10, r.y + 14), (r.right - 12, r.bottom - 12),
                                          (r.x + 12, r.bottom - 12)])


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

STEPPERS = {st.key: st for st in [
    # Qué ve
    ui.Stepper('num_sensors', "Sensores", lo=1, hi=15, step=1,
               help_text="Rayos que miden la distancia a la pared. Más sensores = ve mejor, pero tarda más en aprender."),
    ui.Stepper('sensor_spread', "Ángulo de visión", lo=30, hi=360, step=15, fmt="{}°",
               help_text="Qué tan abiertos están los sensores. 360° = ve para todos lados."),
    ui.Stepper('sensor_range', "Alcance", lo=60, hi=500, step=20, fmt="{} px",
               help_text="Hasta qué distancia llegan los sensores."),
    ui.Stepper('use_speed', "Siente su velocidad",
               help_text="Le pasa su propia velocidad a la red. Ayuda a frenar antes de las curvas."),
    ui.Stepper('use_compass', "Brújula al objetivo",
               help_text="Le dice hacia dónde queda el próximo checkpoint o la salida (en línea recta, sin saber "
                         "de paredes). En el laberinto conviene sacarla: sin brújula aprenden a seguir una pared y "
                         "encuentran la salida de laberintos que nunca vieron."),
    ui.Stepper('see_mates', "Ve a su compañero",
               help_text="Le pasa dónde está el compañero más cercano (3 entradas). Ayuda a no amontonarse."),
    ui.Stepper('see_rivals', "Ve al rival",
               help_text="Le pasa dónde está el rival más cercano (3 entradas). Sirve para esquivarlo o marcarlo."),
    ui.Stepper('use_role', "Sabe su número",
               help_text="Cada jugador sabe si es el 1, el 2... Como todo el equipo usa el mismo cerebro, así "
                         "pueden aprender puestos distintos (uno ataca y otro defiende)."),
    # Cómo piensa
    ui.Stepper('brain_size', "Tamaño de la red", options=[(k, k) for k in BRAIN_SIZES],
               help_text="Neuronas en las capas ocultas: " + ", ".join(f"{k} {'-'.join(map(str, v))}" for k, v in BRAIN_SIZES.items()) + ". Más grande puede aprender más, pero más lento."),
    # Cómo maneja
    ui.Stepper('action_set', "Acciones", options=[(k, ACTION_SHORT[k]) for k in ACTION_SETS],
               help_text=" | ".join(v['label'] for v in ACTION_SETS.values())),
    ui.Stepper('max_speed', "Vel. máxima", lo=2.0, hi=12.0, step=0.5, fmt="{:.1f}",
               help_text="Más rápido = vueltas más cortas, pero más difícil no chocar."),
    ui.Stepper('turn_speed', "Giro", lo=0.03, hi=0.25, step=0.01, fmt="{:.2f}",
               help_text="Cuánto dobla por paso (en radianes). Los laberintos necesitan girar más."),
    # Cómo evoluciona
    ui.Stepper('population', "Autos por gen.", lo=10, hi=300, step=10,
               help_text="Más autos = más variedad para elegir a los mejores, pero cada generación tarda más. "
                         "En fútbol, cada auto de la población es un equipo."),
    ui.Stepper('elite_pct', "Élite", lo=5, hi=50, step=5, fmt="{} %",
               help_text="Porcentaje de los mejores que pasan sin cambios a la siguiente generación."),
    ui.Stepper('mutation_rate', "Mutación: prob.", lo=0.02, hi=0.6, step=0.02, fmt="{:.2f}",
               help_text="Probabilidad de que cada peso de la red cambie al azar en los hijos."),
    ui.Stepper('mutation_strength', "Mutación: fuerza", lo=0.05, hi=1.5, step=0.05, fmt="{:.2f}",
               help_text="Qué tan grande es cada cambio. Grande = explora más; chico = afina."),
    ui.Stepper('crossover', "Cruza entre padres",
               help_text="Los hijos mezclan neuronas de dos padres en vez de copiar a uno solo."),
    ui.Stepper('eval_matches', "Partidos c/u", lo=1, hi=5, step=1,
               help_text="Cuántos partidos juega cada cerebro por generación; su puntaje es el promedio. Más partidos = "
                         "menos suerte en la elección de los mejores, pero cada generación tarda más."),
    ui.Stepper('max_steps', "Pasos por gen.", lo=500, hi=10000, step=250,
               help_text="Tiempo máximo de cada generación."),
    ui.Stepper('patience', "Paciencia", lo=50, hi=1500, step=50,
               help_text="Pasos sin avanzar permitidos. Si un auto no avanza en tantos pasos, queda afuera (evita autos dando "
                         "vueltas en el lugar). En el laberinto, descubrir un pasillo nuevo también cuenta como avanzar."),
    ui.Stepper('bounce', "Rebota al chocar",
               help_text="Sí: al chocar vuelve atrás, frena y sigue, así puede aprender a darse vuelta cuando entra en "
                         "un callejón sin salida. No: el auto que choca queda afuera en esa generación."),
    # Puntaje de pista y laberinto
    ui.Stepper('r_fast', "Premio por rapidez", lo=0.0, hi=10.0, step=0.5, fmt="{:.1f}",
               help_text="Al llegar, suma estos puntos por cada paso que le sobró. Más alto = premia más llegar rápido "
                         "que llegar seguro."),
    ui.Stepper('p_crash', "Castigo por chocar", lo=0, hi=2000, step=100,
               help_text="Puntos que pierde el auto que choca (si rebota, por cada choque). En 0 el choque solo "
                         "lo deja afuera o lo frena."),
    # Puntaje de fútbol
    ui.Stepper('r_goal', "Gol a favor", lo=0, hi=3000, step=100, fmt="+{}",
               help_text="Puntos por cada gol que mete el equipo."),
    ui.Stepper('p_conceded', "Gol en contra", lo=0, hi=3000, step=100, fmt="-{}",
               help_text="Puntos que pierde por cada gol que le hacen."),
    ui.Stepper('p_own_goal', "Gol en propio arco", lo=0, hi=3000, step=100, fmt="-{}",
               help_text="Castigo extra cuando el gol en contra lo hace un jugador del propio equipo."),
    ui.Stepper('r_touch', "Tocar la pelota", lo=0, hi=100, step=5, fmt="+{}",
               help_text="Puntos por cada toque (cuenta hasta 30 por partido). Ayuda al principio, cuando todavía no hay goles."),
    ui.Stepper('r_kick', "Patada útil", lo=0, hi=300, step=10, fmt="+{}",
               help_text="Puntos por cada patada que manda la pelota hacia el arco rival (cuenta hasta 15 por partido). "
                         "No premia patear al aire, para que no aprendan a patear sin parar."),
    ui.Stepper('r_advance', "Llevarla al arco", lo=0.0, hi=5.0, step=0.2, fmt="+{:.1f}",
               help_text="Puntos por cada píxel que la pelota se acerca al arco rival (y se restan si se aleja)."),
    ui.Stepper('r_near', "Estar cerca", lo=0, hi=1000, step=50, fmt="+{}",
               help_text="Premio por estar cerca de la pelota durante el partido. Muy alto = todos persiguen la pelota."),
    ui.Stepper('p_crowd', "Amontonarse", lo=0, hi=1500, step=50, fmt="-{}",
               help_text="Castigo por estar pegado a un compañero (a menos de 70 px). Hace que se repartan la cancha. "
                         "Muy alto = se alejan tanto que nadie va a la pelota."),
    ui.Stepper('p_idle', "Quedarse quieto", lo=0, hi=1500, step=50, fmt="-{}",
               help_text="Castigo por estar quieto (por ejemplo, trabado contra la pared) durante el partido."),
]}


def sections_for(kind):
    """[(título, [claves])] de las opciones que tiene sentido tocar para cada tipo de agente"""
    if kind == 'futbol':
        see = ['num_sensors', 'sensor_spread', 'sensor_range', 'use_speed', 'see_mates', 'see_rivals', 'use_role']
        evo = ['population', 'elite_pct', 'mutation_rate', 'mutation_strength', 'crossover', 'eval_matches']
        score = ['r_goal', 'p_conceded', 'p_own_goal', 'r_touch', 'r_kick', 'r_advance', 'r_near', 'p_crowd', 'p_idle']
    else:
        see = ['num_sensors', 'sensor_spread', 'sensor_range', 'use_speed', 'use_compass']
        evo = ['population', 'elite_pct', 'mutation_rate', 'mutation_strength', 'crossover', 'max_steps', 'patience',
               'bounce']
        score = ['r_fast', 'p_crash']
    return [
        (0, "QUÉ VE", see),
        (0, "CÓMO PIENSA", ['brain_size']),
        (0, "CÓMO MANEJA", ['action_set', 'max_speed', 'turn_speed']),
        (1, "CÓMO EVOLUCIONA", evo),
        (1, "QUÉ PREMIA SU PUNTAJE", score),
    ]


def brain_status(profile, kind, cfg):
    """(existe, compatible, texto)"""
    path = brain_path(profile, kind)
    if not os.path.exists(path):
        return False, False, "Todavía no hay un cerebro entrenado con este perfil."
    try:
        data = np.load(path)
        sizes = list(data['sizes'])
        gen = int(data['generation']) if 'generation' in data else None
        if 'meta' in data:
            gen = json.loads(str(data['meta'])).get('generation', gen)
    except Exception:
        return True, False, "El cerebro guardado está dañado."
    if sizes != cfg.layer_sizes(kind):
        return True, False, ("Hay un cerebro guardado, pero con otra forma de red "
                             f"({'-'.join(map(str, sizes))}). Cambiaste sensores, red o acciones.")
    extra = f" (generación {gen})" if gen else ""
    return True, True, f"Hay un cerebro entrenado{extra}: podés seguir entrenándolo."


class Workshop:
    def __init__(self, screen, scenario, cfg, profile):
        self.screen = screen
        self.fonts = ui.Fonts()
        self.scenario = scenario
        self.kind = scenario.key
        self.cfg = copy_config(cfg)
        self.cfg.kind = self.kind
        self.name = ui.TextInput(profile)
        self.editing_name = False
        self.toast = ("", 0)
        self.sections = sections_for(self.kind)
        self.steppers = [STEPPERS[k] for _, _, keys in self.sections for k in keys]

    # -------------------------------------------------------------- #
    def layout(self):
        W, H = self.screen.get_size()
        col_w = min(340, (W - 500) // 2)
        self.cols = [pygame.Rect(30, 112, col_w, 0), pygame.Rect(30 + col_w + 18, 112, col_w, 0)]
        n_rows = [0, 0]
        n_heads = [0, 0]
        for col, _, keys in self.sections:
            n_rows[col] += len(keys)
            n_heads[col] += 1
        avail = H - 100 - 112
        row_h = max(24, min(31, min((avail - n_heads[c] * 30) // max(1, n_rows[c]) for c in (0, 1))))
        ys = [self.cols[0].y, self.cols[1].y]
        self.headers = []
        for col, title, keys in self.sections:
            x = self.cols[col].x
            self.headers.append((title, (x, ys[col])))
            ys[col] += 24
            for k in keys:
                STEPPERS[k].layout(x, ys[col], col_w, row_h - 3)
                ys[col] += row_h
            ys[col] += 6

        # Derecha: perfil, presets, vista previa
        right = pygame.Rect(self.cols[1].right + 24, 70, W - self.cols[1].right - 48, H - 160)
        self.right = right
        y = right.y
        self.profile_y = y
        self.swatch = pygame.Rect(right.x, y + 24, 54, 30)
        self.name_rect = pygame.Rect(self.swatch.right + 8, y + 24, right.w - 54 - 8 - 100, 30)
        self.save_rect = pygame.Rect(self.name_rect.right + 6, y + 24, 94, 30)
        y += 60
        profiles = [p for p in list_profiles(self.kind) if not p.startswith('_')]
        self.profile_rects = []
        x, rows = right.x, 0
        for p in profiles:
            w = min(right.w, self.fonts.tiny.size(p)[0] + 18)
            if x + w > right.right:
                x, rows = right.x, rows + 1
                y += 28
            if rows >= 2:
                break
            self.profile_rects.append((p, pygame.Rect(x, y, w, 24)))
            x += w + 6
        y += 32 if profiles else 4
        self.presets_y = y
        names = list(PRESETS[self.kind])
        pw = (right.w - 6) // 2
        self.preset_rects = [(name, pygame.Rect(right.x + (i % 2) * (pw + 6), y + 24 + (i // 2) * 32, pw, 27))
                             for i, name in enumerate(names)]
        y += 24 + ((len(names) + 1) // 2) * 32 + 8
        self.preview_rect = pygame.Rect(right.x, y + 22, right.w, right.bottom - y - 22 - 60)
        self.net_y = self.preview_rect.bottom + 8
        self.btn_detail = pygame.Rect(right.x, self.net_y + 24, right.w, 30)

        bw = 230
        self.btn_back = pygame.Rect(30, H - 74, 180, 48)
        self.btn_new = pygame.Rect(W - 2 * bw - 40, H - 74, bw, 48)
        self.btn_continue = pygame.Rect(W - bw - 30, H - 74, bw, 48)

    # -------------------------------------------------------------- #
    def flash(self, msg):
        self.toast = (msg, pygame.time.get_ticks() + 2500)

    def open_detail(self):
        brain = read_brain(brain_path(self.profile, self.kind))
        return agent_detail(self.screen, self.cfg, self.kind, brain,
                            f"Perfil \"{self.profile}\" · {KIND_LABELS[self.kind]}")

    def new_color(self):
        self.cfg.color = random_color(used_colors() + [self.cfg.color])

    def handle_click(self, pos):
        self.editing_name = self.name_rect.collidepoint(pos)
        for st in self.steppers:
            value = getattr(self.cfg, st.key)
            new = st.click(pos, value)
            if new is not None:
                setattr(self.cfg, st.key, new)
                return None
        for name, r in self.preset_rects:
            if r.collidepoint(pos):
                self.cfg = preset(self.kind, name, color=self.cfg.color)
                self.flash(f"Preset \"{name}\" aplicado")
                return None
        if self.swatch.collidepoint(pos):
            self.new_color()
        if self.save_rect.collidepoint(pos):
            self.save_profile()
        for p, r in self.profile_rects:
            if r.collidepoint(pos):
                try:
                    self.cfg = AgentConfig.load(self.kind, p)
                    self.name.value = p
                    self.flash(f"Perfil \"{p}\" cargado")
                except Exception:
                    self.flash("No se pudo cargar ese perfil")
        if self.btn_detail.collidepoint(pos):
            return 'detail'
        if self.btn_back.collidepoint(pos):
            return 'back'
        if self.btn_new.collidepoint(pos):
            return 'new'
        if self.btn_continue.collidepoint(pos) and brain_status(self.profile, self.kind, self.cfg)[1]:
            return 'continue'
        return None

    @property
    def profile(self):
        name = "".join(c for c in self.name.value.strip() if c.isalnum() or c in "-_ ").strip()
        return name or "perfil"

    def ensure_own_color(self):
        """Un perfil nuevo no puede tener el mismo color que otro ya guardado"""
        if self.profile not in list_profiles(self.kind) and self.cfg.color in used_colors():
            self.new_color()

    def save_profile(self):
        self.ensure_own_color()
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

        reach = self.cfg.sensor_range + 40
        src = pygame.Rect(int(sx - reach), int(sy - reach), 2 * reach, 2 * reach)
        crop = pygame.Surface(src.size)
        crop.fill(ui.BG)
        crop.blit(sc.background, (0, 0), src)
        side = max(10, min(rect.w, rect.h) - 12)
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
        draw_car(self.screen, car, sa, s, self.cfg.color)

    def draw(self):
        f, scr = self.fonts, self.screen
        W, H = scr.get_size()
        self.layout()
        scr.fill(ui.BG)
        ui.text(scr, f"TALLER DEL AGENTE DE {KIND_LABELS[self.kind].upper()}", f.title, ui.YELLOW, topleft=(30, 22))
        ui.text(scr, ui.fit(f"Escenario: {self.scenario.title}  ·  {getattr(self.scenario, 'name', '') or self.scenario.goal_text}",
                            f.small, self.right.x - 50), f.small, ui.TEXT_DIM, topleft=(32, 64))
        ui.text(scr, "Pasá el mouse sobre una opción para ver qué hace", f.tiny, ui.TEXT_DIM, topleft=(32, 86))

        mouse = pygame.mouse.get_pos()
        hover = None
        for title, pos in self.headers:
            ui.text(scr, title, f.small, ui.CYAN, topleft=(pos[0] + 2, pos[1] + 4))
        for st in self.steppers:
            h = st.rect.collidepoint(mouse)
            st.draw(scr, f, getattr(self.cfg, st.key), highlight=h)
            if h:
                hover = st.help

        # Perfil: color, nombre y perfiles guardados de este tipo
        ui.text(scr, "PERFIL (color: clic en el auto)", f.small, ui.CYAN, topleft=(self.right.x + 2, self.profile_y + 4))
        pygame.draw.rect(scr, ui.PANEL_LIGHT if self.swatch.collidepoint(mouse) else ui.PANEL, self.swatch, border_radius=6)
        draw_car(scr, self.swatch.center, 0, 1.1, self.cfg.color)
        if self.swatch.collidepoint(mouse):
            hover = "Clic para darle otro color al azar. El color sirve para reconocer al agente en la galería y en el modo Expo."
        pygame.draw.rect(scr, (28, 30, 40), self.name_rect, border_radius=6)
        pygame.draw.rect(scr, ui.YELLOW if self.editing_name else ui.PANEL_LIGHT, self.name_rect, 2, border_radius=6)
        cursor = "|" if self.editing_name and (pygame.time.get_ticks() // 500) % 2 else ""
        ui.text(scr, ui.fit(self.name.value + cursor, f.small, self.name_rect.w - 12), f.small, ui.WHITE,
                midleft=(self.name_rect.x + 8, self.name_rect.centery))
        ui.button(scr, f.small, "Guardar", self.save_rect)
        for p, r in self.profile_rects:
            ui.button(scr, f.tiny, p, r, active=(p == self.profile))

        ui.text(scr, f"PRESETS DE {KIND_LABELS[self.kind].upper()}", f.small, ui.CYAN,
                topleft=(self.right.x + 2, self.presets_y + 4))
        for name, r in self.preset_rects:
            ui.button(scr, f.tiny, name, r)
            if r.collidepoint(mouse):
                hover = PRESET_HELP.get((self.kind, name), "")

        ui.text(scr, "Lo que ve en la salida", f.small, ui.WHITE, topleft=(self.right.x, self.preview_rect.y - 20))
        self.draw_preview(self.preview_rect)
        sizes = self.cfg.layer_sizes(self.kind)
        ui.text(scr, ui.fit(f"Su red: {' - '.join(map(str, sizes))}  ({self.cfg.num_params(self.kind)} pesos)",
                            f.small, self.right.w), f.small, ui.WHITE, topleft=(self.right.x, self.net_y))
        ui.button(scr, f.small, "Detalle del agente: qué hace cada neurona (D)", self.btn_detail, ui.PURPLE)

        # Ayuda contextual
        help_rect = pygame.Rect(self.btn_back.right + 20, H - 78, self.btn_new.x - self.btn_back.right - 40, 56)
        exists, ok, status = brain_status(self.profile, self.kind, self.cfg)
        if hover:
            _wrap(scr, hover, f.small, ui.WHITE, help_rect, 20)
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
                    if event.key == pygame.K_d and self.open_detail() == 'quit':
                        return 'quit'
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    result = self.handle_click(event.pos)
                    if result == 'detail':
                        if self.open_detail() == 'quit':
                            return 'quit'
                    elif result:
                        return result
            self.draw()
            clock.tick(60)


PRESET_HELP = {
    ('pista', "Equilibrado"): "5 sensores, red mediana, 40 autos. Sirve para casi cualquier pista.",
    ('pista', "Rápido de entrenar"): "3 sensores, red chica y siempre acelera: aprende en pocas generaciones, ideal para mostrar en vivo.",
    ('pista', "Explorador"): "80 autos con mutaciones grandes: prueba muchas ideas distintas. Bueno para pistas difíciles.",
    ('pista', "Preciso"): "9 sensores, red grande y puede frenar: maneja más fino, pero tarda más en aprender.",
    ('laberinto', "Sigue paredes"): "Sin brújula: aprende a seguir una pared hasta la salida. Es el que mejor resuelve laberintos nuevos.",
    ('laberinto', "Explorador"): "Con brújula a la salida, 7 sensores a 180° y gira más fuerte. Puede meterse en callejones que apuntan a la salida.",
    ('laberinto', "Rápido de entrenar"): "Red chica y siempre acelera. Para laberintos chicos.",
    ('laberinto', "Generalista"): "9 sensores a 270° y red grande: pensado para 'laberinto nuevo cada N generaciones'.",
    ('futbol', "Goleador"): "Los puntajes de siempre: premia goles, toques y llevar la pelota al arco.",
    ('futbol', "Equipo ordenado"): "Castiga mucho amontonarse y premia menos estar cerca de la pelota: para equipos de 2 o más.",
    ('futbol', "Defensor"): "Castiga el doble los goles en contra: aprende a cuidar su arco antes de atacar.",
    ('futbol', "Rápido de entrenar"): "3 sensores, red chica y no ve al rival: aprende más rápido sin rival o contra el bot fácil.",
}


def workshop(screen, scenario, cfg, profile):
    """Devuelve (acción, config, perfil). acción: 'new' | 'continue' | 'back' | 'quit'"""
    ws = Workshop(screen, scenario, cfg, profile)
    action = ws.run()
    if action in ('new', 'continue'):
        ws.ensure_own_color()
        ws.cfg.save(ws.profile)
    return action, ws.cfg, ws.profile


# ---------------------------------------------------------------------- #
# Varias pistas: elegir cuáles y en qué orden
# ---------------------------------------------------------------------- #
CURRICULUM_STEPPERS = [
    ui.Stepper('rule', "Pasar a la siguiente", options=[('dominar', "dominarla"), ('cada', "cada N")],
               help_text="Al dominarla: cuando cierto porcentaje de autos completa la pista. "
                         "Cada N gen.: cambia de pista cada tantas generaciones."),
    ui.Stepper('threshold', "Dominarla = llegan", lo=10, hi=100, step=10, fmt="{} %",
               help_text="Porcentaje de autos que tiene que completar la pista para pasar a la siguiente."),
    ui.Stepper('every', "N generaciones", lo=5, hi=100, step=5,
               help_text="Generaciones en cada pista cuando se cambia cada N."),
]


def list_tracks():
    """[(archivo, nombre visible, miniatura o None)] de las pistas que no son canchas"""
    import track_selector as ts
    out = []
    if not os.path.isdir(ts.TRACKS_DIR):
        return out
    for f in sorted(os.listdir(ts.TRACKS_DIR)):
        if not f.endswith('.json'):
            continue
        name = f[:-5]
        try:
            with open(os.path.join(ts.TRACKS_DIR, f)) as fh:
                meta = json.load(fh)
        except Exception:
            continue
        if ts.is_field(meta):
            continue
        thumb = None
        for suffix in ('_thumb.png', '.png'):
            path = os.path.join(ts.TRACKS_DIR, name + suffix)
            if os.path.exists(path):
                try:
                    thumb = pygame.image.load(path)
                except Exception:
                    pass
                break
        out.append((name, ts.display_name(meta, name), thumb))
    return out


def curriculum_settings(screen, opts):
    """Devuelve la lista ordenada de pistas elegidas, None (volver) o 'quit'. opts se modifica."""
    fonts = ui.Fonts()
    clock = pygame.time.Clock()
    tracks = list_tracks()
    thumbs = {}
    chosen = [t for t in opts.get('tracks', []) if t in {n for n, _, _ in tracks}]
    scroll = 0
    while True:
        W, H = screen.get_size()
        grid = pygame.Rect(30, 110, W - 470, H - 200)
        cols = max(2, grid.w // 200)
        tw = (grid.w - (cols - 1) * 12) // cols
        th = int(tw * 0.62) + 26
        rows = (len(tracks) + cols - 1) // cols
        max_scroll = max(0, rows * (th + 12) - grid.h)
        scroll = max(0, min(scroll, max_scroll))
        tiles = [(name, pygame.Rect(grid.x + (i % cols) * (tw + 12), grid.y + (i // cols) * (th + 12) - scroll, tw, th))
                 for i, (name, _, _) in enumerate(tracks)]
        side = pygame.Rect(grid.right + 30, 110, W - grid.right - 60, H - 200)
        for i, st in enumerate(CURRICULUM_STEPPERS):
            st.layout(side.x, side.y + 30 + i * 44, side.w, 36)
        back = pygame.Rect(30, H - 74, 180, 48)
        go = pygame.Rect(W - 290, H - 74, 260, 48)
        clear = pygame.Rect(side.x, side.y + 30 + len(CURRICULUM_STEPPERS) * 44 + 10, side.w, 34)

        for event in pygame.event.get():
            if _quit_event(event):
                return 'quit'
            if event.type == pygame.MOUSEWHEEL:
                scroll -= event.y * 40
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN and len(chosen) >= 1:
                    opts['tracks'] = chosen
                    return list(chosen)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for name, r in tiles:
                    if r.collidepoint(event.pos) and grid.collidepoint(event.pos):
                        if name in chosen:
                            chosen.remove(name)
                        else:
                            chosen.append(name)
                for st in CURRICULUM_STEPPERS:
                    new = st.click(event.pos, opts[st.key])
                    if new is not None:
                        opts[st.key] = new
                if clear.collidepoint(event.pos):
                    chosen = []
                if back.collidepoint(event.pos):
                    return None
                if go.collidepoint(event.pos) and chosen:
                    opts['tracks'] = chosen
                    return list(chosen)

        screen.fill(ui.BG)
        ui.text(screen, "VARIAS PISTAS", fonts.title, ui.YELLOW, topleft=(30, 24))
        ui.text(screen, "Elegí las pistas en el orden en que las va a aprender (clic para sumar o sacar)",
                fonts.small, ui.TEXT_DIM, topleft=(32, 70))
        screen.set_clip(grid)
        if not tracks:
            ui.text(screen, "No hay pistas guardadas. Creá algunas con el editor.", fonts.normal, ui.TEXT_DIM,
                    center=grid.center)
        for (name, title, thumb), (_, r) in zip(tracks, tiles):
            if r.bottom < grid.y or r.y > grid.bottom:
                continue
            on = name in chosen
            pygame.draw.rect(screen, ui.PANEL_LIGHT if on else ui.PANEL, r, border_radius=10)
            img_r = pygame.Rect(r.x + 6, r.y + 6, r.w - 12, r.h - 34)
            if thumb is not None:
                if name not in thumbs:
                    thumbs[name] = pygame.transform.smoothscale(thumb, img_r.size)
                screen.blit(thumbs[name], img_r)
            ui.text(screen, ui.fit(title, fonts.tiny, r.w - 12), fonts.tiny, ui.WHITE, midbottom=(r.centerx, r.bottom - 6))
            if on:
                pygame.draw.rect(screen, ui.YELLOW, r, 3, border_radius=10)
                badge = pygame.Rect(r.x + 10, r.y + 10, 30, 30)
                pygame.draw.circle(screen, ui.YELLOW, badge.center, 15)
                ui.text(screen, str(chosen.index(name) + 1), fonts.normal, ui.BLACK, center=badge.center)
        screen.set_clip(None)

        ui.text(screen, "CUÁNDO CAMBIA DE PISTA", fonts.small, ui.CYAN, topleft=(side.x, side.y))
        mouse = pygame.mouse.get_pos()
        hover = None
        for st in CURRICULUM_STEPPERS:
            h = st.rect.collidepoint(mouse)
            dim = (st.key == 'threshold' and opts['rule'] == 'cada') or (st.key == 'every' and opts['rule'] == 'dominar')
            if not dim:
                st.draw(screen, fonts, opts[st.key], highlight=h)
            if h:
                hover = st
        ui.button(screen, fonts.small, "Sacar todas", clear)
        y = clear.bottom + 20
        ui.text(screen, f"Elegidas: {len(chosen)}", fonts.normal, ui.WHITE, topleft=(side.x, y))
        y += 30
        names = dict((n, t) for n, t, _ in tracks)
        for i, name in enumerate(chosen[:10]):
            ui.text(screen, ui.fit(f"{i + 1}. {names.get(name, name)}", fonts.small, side.w), fonts.small, ui.TEXT_DIM,
                    topleft=(side.x, y))
            y += 22
        y += 10
        help_txt = hover.help if hover else ("Consejo: empezá con pistas fáciles (un óvalo) y terminá con las difíciles. "
                                             "Al final de la lista vuelve a la primera.")
        _wrap(screen, help_txt, fonts.small, ui.WHITE if hover else ui.TEXT_DIM, pygame.Rect(side.x, y, side.w, 120), 20)
        ui.button(screen, fonts.normal, "Volver (Esc)", back)
        ui.button(screen, fonts.normal, "Siguiente (Enter)", go, ui.GREEN, enabled=bool(chosen))
        pygame.display.flip()
        clock.tick(60)


# ---------------------------------------------------------------------- #
# Fútbol: opciones del partido
# ---------------------------------------------------------------------- #
from ai.football import OPPONENTS, BALL_CONTROLS, FootballScenario, TEAM_COLORS  # noqa: E402

FOOTBALL_STEPPERS = [
    ui.Stepper('team_size', "Jugadores por equipo", options=[(n, str(n)) for n in range(1, 7)],
               help_text="Con más jugadores tienen que aprender a no estorbarse y cada partido es más lento. "
                         "Conviene empezar con 1 y subir de a poco. Con 4 o más arman dos líneas."),
    ui.Stepper('opponent', "Rival", options=OPPONENTS,
               help_text="Sin rival: practican goles (lo más fácil para empezar). Bot: un auto programado que "
                         "va a la pelota. Entre ellos: los cerebros juegan partidos entre sí."),
    ui.Stepper('ball_control', "Control de pelota", options=BALL_CONTROLS,
               help_text="Pegada: al tocarla de frente queda pegada al auto hasta que patea, choca contra la pared "
                         "o se la toca un rival (más fácil de aprender). Libre: la pelota se mueve solo con la física."),
    ui.Stepper('random_starts', "Saques al azar", options=[(0.0, "No"), (0.5, "La mitad"), (1.0, "Todos")],
               help_text="Algunos partidos empiezan con la pelota y los autos en lugares al azar. Así no se "
                         "aprenden de memoria un solo saque y miran de verdad hacia qué arco van. La mitad anda mejor."),
    ui.Stepper('match_steps', "Duración del partido",
               options=[(900, "15 s"), (1200, "20 s"), (1800, "30 s"), (2700, "45 s"), (3600, "1 min")],
               help_text="Tiempo de cada partido (en tiempo de juego, a velocidad x1). Cada generación es un partido."),
]


def football_settings(screen, opts, field):
    """Devuelve ('go', escenario), ('pick', None), ('back', None) o ('quit', None)"""
    fonts = ui.Fonts()
    clock = pygame.time.Clock()
    preview, cache_key = None, None
    while True:
        W, H = screen.get_size()
        left = pygame.Rect(40, 110, 430, H - 220)
        for i, st in enumerate(FOOTBALL_STEPPERS):
            st.layout(left.x, left.y + 40 + i * 48, left.w, 38)
        field_rect = pygame.Rect(left.x, left.y, left.w, 32)
        pick = pygame.Rect(left.x, left.y + 40 + len(FOOTBALL_STEPPERS) * 48 + 10, (left.w - 8) // 2, 40)
        classic = pygame.Rect(pick.right + 8, pick.y, (left.w - 8) // 2, 40)
        back = pygame.Rect(40, H - 80, 200, 50)
        go = pygame.Rect(W - 300, H - 80, 260, 50)
        area = pygame.Rect(left.right + 40, 110, W - left.right - 80, H - 220)

        for event in pygame.event.get():
            if _quit_event(event):
                return 'quit', None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'back', None
                if event.key == pygame.K_RETURN:
                    return 'go', FootballScenario(field, opts['team_size'], opts['opponent'], opts['match_steps'],
                                                 opts['ball_control'], opts.get('random_starts', 0.0))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for st in FOOTBALL_STEPPERS:
                    new = st.click(event.pos, opts[st.key])
                    if new is not None:
                        opts[st.key] = new
                if pick.collidepoint(event.pos):
                    return 'pick', None
                if classic.collidepoint(event.pos) and field.file:
                    return 'classic', None
                if back.collidepoint(event.pos):
                    return 'back', None
                if go.collidepoint(event.pos):
                    return 'go', FootballScenario(field, opts['team_size'], opts['opponent'], opts['match_steps'],
                                                 opts['ball_control'], opts.get('random_starts', 0.0))

        key = (id(field), area.size)
        if key != cache_key:
            mw, mh = field.size
            s = min(area.w / mw, area.h / mh)
            preview = pygame.transform.smoothscale(field.background, (int(mw * s), int(mh * s)))
            cache_key = key

        screen.fill(ui.BG)
        ui.text(screen, "FÚTBOL", fonts.title, ui.YELLOW, topleft=(40, 30))
        ui.text(screen, "Cada cerebro maneja a todos los jugadores de su equipo (azul). Ganan puntos por "
                "goles, toques y por acercar la pelota al arco rival.", fonts.small, ui.TEXT_DIM, topleft=(42, 74))
        ui.text(screen, ui.fit(f"Cancha: {field.name}", fonts.normal, left.w), fonts.normal, ui.WHITE,
                midleft=(field_rect.x + 4, field_rect.centery))
        mouse = pygame.mouse.get_pos()
        hover = None
        for st in FOOTBALL_STEPPERS:
            h = st.rect.collidepoint(mouse)
            st.draw(screen, fonts, opts[st.key], highlight=h)
            if h:
                hover = st
        ui.button(screen, fonts.small, "Elegir cancha del editor", pick)
        ui.button(screen, fonts.small, "Usar la clásica", classic, enabled=bool(field.file))
        tip = hover.help if hover else ("Consejo: con la pelota pegada aprenden mucho más rápido. Entrená primero "
                                        "sin rival hasta que metan goles y después seguí con el mismo perfil contra el "
                                        "bot: ahí aprenden a patear, porque el rival les roba la pelota.")
        _wrap(screen, tip, fonts.small, ui.WHITE if hover else ui.TEXT_DIM,
              pygame.Rect(left.x, pick.bottom + 24, left.w, 140), 20)

        r = preview.get_rect(center=area.center)
        screen.blit(preview, r)
        s = r.w / field.size[0]
        teams = 2 if opts['opponent'] != 'none' else 1
        for t in range(teams):
            for x, y, a in field.team_spawns(t, opts['team_size']):
                pygame.draw.circle(screen, TEAM_COLORS[t], (int(r.x + x * s), int(r.y + y * s)), 9)
                pygame.draw.circle(screen, ui.WHITE, (int(r.x + x * s), int(r.y + y * s)), 9, 2)
        bx, by = field.ball_spawn
        pygame.draw.circle(screen, ui.WHITE, (int(r.x + bx * s), int(r.y + by * s)), 6)

        ui.button(screen, fonts.normal, "Volver (Esc)", back)
        ui.button(screen, fonts.normal, "Siguiente (Enter)", go, ui.GREEN)
        pygame.display.flip()
        clock.tick(60)


# ---------------------------------------------------------------------- #
# Campeones: galería con modo Expo y jugar contra la IA
# ---------------------------------------------------------------------- #
def gallery(screen):
    """Devuelve (acción, entrada, otra_pista): acción 'expo' | 'competir' | 'back' | 'quit'"""
    from ai import hall
    fonts = ui.Fonts()
    clock = pygame.time.Clock()
    entries = hall.list_entries()
    sel, scroll = 0, 0
    confirm_delete = False
    other_map = False
    toast = ("", 0)
    while True:
        W, H = screen.get_size()
        lst = pygame.Rect(30, 100, min(520, W // 2 - 40), H - 130)
        row_h = 62
        max_scroll = max(0, len(entries) * row_h - lst.h)
        scroll = max(0, min(scroll, max_scroll))
        rows = [pygame.Rect(lst.x, lst.y + i * row_h - scroll, lst.w, row_h - 6) for i in range(len(entries))]
        det = pygame.Rect(lst.right + 30, 100, W - lst.right - 60, H - 130)
        entry = entries[sel] if entries else None
        bw = (det.w - 50) // 2
        by = det.bottom - 116
        buttons = []
        if entry:
            bx = det.x + 20
            buttons.append(('expo', "Modo Expo (E)", pygame.Rect(bx, by, bw, 46), ui.GREEN))
            buttons.append(('competir', "Jugar contra él (J)", pygame.Rect(bx + bw + 10, by, bw, 46), ui.ORANGE))
            tw = (det.w - 60) // 3
            row2 = [pygame.Rect(bx + i * (tw + 10), by + 56, tw, 38) for i in range(3)]
            buttons.append(('detail', "Detalle (D)", row2[0], ui.PURPLE))
            if entry['kind'] == 'pista':
                buttons.append(('other', "Pista: elegir otra" if other_map else "Pista: la suya", row2[1],
                                ui.BLUE if other_map else ui.PANEL_LIGHT))
            if entry['gallery']:
                buttons.append(('delete', "¿Seguro? Clic otra vez" if confirm_delete else "Eliminar",
                                row2[2], ui.RED if confirm_delete else ui.PANEL_LIGHT))
            else:
                buttons.append(('promote', "A la galería", row2[2], ui.PANEL_LIGHT))

        def go(action):
            return action, entry, other_map and entry['kind'] == 'pista'

        for event in pygame.event.get():
            if _quit_event(event):
                return 'quit', None, False
            if event.type == pygame.MOUSEWHEEL:
                scroll -= event.y * 40
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'back', None, False
                if entries and event.key in (pygame.K_UP, pygame.K_DOWN):
                    sel = max(0, min(len(entries) - 1, sel + (1 if event.key == pygame.K_DOWN else -1)))
                    confirm_delete = False
                if entry and event.key in (pygame.K_RETURN, pygame.K_e):
                    return go('expo')
                if entry and event.key == pygame.K_j:
                    return go('competir')
                if entry and event.key == pygame.K_d:
                    if _entry_detail(screen, entry) == 'quit':
                        return 'quit', None, False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(rows):
                    if r.collidepoint(event.pos) and lst.collidepoint(event.pos):
                        if i != sel:
                            confirm_delete = False
                        sel = i
                for action, _, r, _ in buttons:
                    if not r.collidepoint(event.pos):
                        continue
                    if action in ('expo', 'competir'):
                        return go(action)
                    if action == 'detail' and _entry_detail(screen, entry) == 'quit':
                        return 'quit', None, False
                    if action == 'other':
                        other_map = not other_map
                    if action == 'promote':
                        name = hall.promote(entry)
                        entries = hall.list_entries()
                        toast = (f"Guardado: {name}", pygame.time.get_ticks() + 2500)
                    if action == 'delete':
                        if confirm_delete:
                            hall.delete_entry(entry)
                            entries = hall.list_entries()
                            sel = max(0, min(sel, len(entries) - 1))
                            confirm_delete = False
                        else:
                            confirm_delete = True

        screen.fill(ui.BG)
        ui.text(screen, "CAMPEONES", fonts.big, ui.YELLOW, topleft=(30, 26))
        ui.text(screen, "Con punto amarillo, los campeones guardados. Sin punto, el último cerebro de cada perfil.",
                fonts.small, ui.TEXT_DIM, topleft=(32, 62))
        if not entries:
            ui.text(screen, "Todavía no hay cerebros entrenados. Entrená uno y volvé.", fonts.normal, ui.TEXT_DIM,
                    center=(W // 2, H // 2))
        screen.set_clip(lst)
        for i, (e, r) in enumerate(zip(entries, rows)):
            pygame.draw.rect(screen, ui.PANEL_LIGHT if i == sel else ui.PANEL, r, border_radius=8)
            if i == sel:
                pygame.draw.rect(screen, ui.YELLOW, r, 2, border_radius=8)
            color = {'pista': ui.BLUE, 'laberinto': ui.PURPLE, 'futbol': (40, 150, 70)}.get(e['kind'], ui.PANEL_LIGHT)
            pygame.draw.rect(screen, color, (r.x, r.y, 8, r.h), border_radius=4)
            draw_car(screen, (r.right - 34, r.centery), 0, 1.0, e['cfg'].color)
            tx = r.x + 18
            if e['gallery']:
                pygame.draw.circle(screen, ui.YELLOW, (tx + 6, r.y + 16), 6)
                tx += 18
            ui.text(screen, ui.fit(e['name'], fonts.small, r.right - tx - 70), fonts.small, ui.WHITE, topleft=(tx, r.y + 8))
            m = e['meta']
            sub = f"Gen. {m.get('generation', '?')}  ·  {m.get('map', '')}  ·  {m.get('date', '')}"
            ui.text(screen, ui.fit(sub, fonts.tiny, r.w - 90), fonts.tiny, ui.TEXT_DIM, topleft=(r.x + 18, r.y + 32))
        screen.set_clip(None)

        mouse = pygame.mouse.get_pos()
        if entry:
            m = entry['meta']
            pygame.draw.rect(screen, ui.PANEL, det, border_radius=12)
            x, y = det.x + 20, det.y + 16
            draw_car(screen, (det.right - 60, det.y + 34), 0, 1.8, entry['cfg'].color)
            ui.text(screen, ui.fit(entry['name'], fonts.big, det.w - 140), fonts.big, ui.WHITE, topleft=(x, y))
            y += 44
            info = [("Escenario", KIND_LABELS.get(entry['kind'], entry['kind'])), ("Mapa", m.get('map', '-')),
                    ("Generaciones", str(m.get('generation', '?'))), ("Puntaje", f"{m.get('fitness', 0):.0f}"),
                    ("Perfil", m.get('profile', '-')), ("Fecha", m.get('date', '-')),
                    ("Gen. 1 guardada", "Sí" if entry['novice'] else "No (se usa una al azar)")]
            if entry['kind'] == 'futbol':
                info.append(("Jugadores", f"{m.get('team_size', 1)} por equipo"))
            for label, value in info:
                ui.text(screen, label, fonts.small, ui.TEXT_DIM, topleft=(x, y))
                ui.text(screen, ui.fit(str(value), fonts.small, det.w - 200), fonts.small, ui.WHITE, topleft=(x + 170, y))
                y += 24
            net = pygame.Rect(x, y + 10, det.w - 40, by - y - 54)
            if net.h > 60:
                ui.draw_network(screen, fonts, net, entry['sizes'])
            tips = {'expo': "Pantalla dividida: un auto de la generación 1 contra este campeón, en el mismo mapa.",
                    'competir': "Manejás vos con las flechas (y espacio para patear en fútbol) contra este cerebro.",
                    'detail': "Qué significa cada neurona de entrada y de salida, y cuánto usa cada una.",
                    'other': "Probalo en una pista que nunca vio: si maneja bien, aprendió a manejar y no a memorizar."}
            tip = next((tips.get(a, "") for a, _, r, _ in buttons if r.collidepoint(mouse)), "")
            if tip:
                _wrap(screen, tip, fonts.tiny, ui.TEXT_DIM, pygame.Rect(x, by - 40, det.w - 40, 40), 17)
            for action, label, r, color in buttons:
                ui.button(screen, fonts.normal if r.h > 42 else fonts.small, label, r, color)
        msg, until = toast
        if msg and pygame.time.get_ticks() < until:
            ui.text(screen, msg, fonts.small, ui.GREEN, topright=(W - 30, 30))
        pygame.display.flip()
        clock.tick(60)


def _entry_detail(screen, entry):
    return agent_detail(screen, entry['cfg'], entry['kind'], entry['brain'], entry['name'])
