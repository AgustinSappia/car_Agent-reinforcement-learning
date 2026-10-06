"""
Modos para mostrar cerebros ya entrenados (pensados para la Expo):

- ExpoScreen: pantalla dividida. Izquierda, un auto de la generación 1; derecha, el campeón.
  En fútbol juegan un partido entre ellos.
- RaceScreen: jugá contra la IA manejando con las flechas. En fútbol, jugás un partido.
"""

import copy
import math

import numpy as np
import pygame

from ai import ui
from ai.brain import PopulationBrain, single_brain
from ai.football import FootballSim, draw_match, draw_score, TEAM_COLORS, KICK_COOLDOWN
from ai.sprites import draw_car, faded
from ai.world import World, observe

PLAYER_COLOR = (80, 170, 255)


def novice_brain(entry):
    """Cerebro de la generación 1 (si se guardó) o uno al azar del mismo tamaño"""
    if entry['novice'] is not None and entry['novice']['sizes'] == entry['sizes']:
        return single_brain(entry['novice']), "Generación 1 (auto típico)"
    return PopulationBrain(entry['sizes'], 1), "Generación 1 (al azar)"


def read_keys():
    k = pygame.key.get_pressed()
    steer = (1 if (k[pygame.K_RIGHT] or k[pygame.K_d]) else 0) - (1 if (k[pygame.K_LEFT] or k[pygame.K_a]) else 0)
    throttle = 1 if (k[pygame.K_UP] or k[pygame.K_w]) else 0
    brake = 1 if (k[pygame.K_DOWN] or k[pygame.K_s]) else 0
    return steer, throttle, brake


class MapView:
    """Un mapa escalado para entrar en un rectángulo de la pantalla"""

    def __init__(self, background, rect):
        mw, mh = background.get_size()
        self.scale = min(rect.w / mw, rect.h / mh)
        w, h = int(mw * self.scale), int(mh * self.scale)
        self.rect = pygame.Rect(rect.x + (rect.w - w) // 2, rect.y + (rect.h - h) // 2, w, h)
        self.bg = pygame.transform.smoothscale(background, (w, h))

    def to_screen(self, p):
        return (float(self.rect.x + p[0] * self.scale), float(self.rect.y + p[1] * self.scale))


# ---------------------------------------------------------------------- #
# Un auto solo en una pista o laberinto
# ---------------------------------------------------------------------- #
class SoloRun:
    def __init__(self, scenario, cfg, brain=None, bounce=False):
        self.sc = copy.copy(scenario)
        self.cfg = cfg
        self.brain = brain          # None = lo maneja una persona
        self.bounce = bounce        # True: al chocar rebota en vez de quedar afuera
        self.world = World(self.sc.road_mask, cfg, self.sc.speed_mask, self.sc.slow_mask, scenario.key)
        self.reset()

    def reset(self):
        self.sc.start_generation(1, 1)
        x, y, a = self.sc.spawn
        self.world.reset(1, x, y, a)
        self.step_n = 0
        self.state = 'run'          # run | done | crash | stuck
        self.trail = []
        self.bumps = 0

    @property
    def pos(self):
        return float(self.world.x[0]), float(self.world.y[0])

    @property
    def angle(self):
        return float(self.world.angle[0])

    def step(self, controls=None):
        if self.state != 'run':
            return
        w, idx = self.world, np.array([0])
        px, py = w.x[idx].copy(), w.y[idx].copy()
        if self.brain is None:
            steer, throttle, brake = controls
            crashed = w.step_raw(idx, np.array([steer], np.float32), np.array([throttle], np.float32),
                                 np.array([brake], np.float32))
        else:
            action = self.brain.forward(observe(w, self.sc, self.cfg, idx), np.array([0]))
            crashed = w.step(idx, action)
        if crashed[0]:
            if self.bounce:
                w.x[idx], w.y[idx] = px, py
                w.speed[idx] = 0
                self.bumps += 1
            else:
                self.state = 'crash'
                return
        done = self.sc.update(w, idx, px, py, self.step_n)
        self.step_n += 1
        if self.step_n % 4 == 0:
            self.trail.append(self.pos)
        if done[0]:
            self.state = 'done'
        elif self.brain is not None and self.step_n - self.sc.last_improve[0] > self.cfg.patience:
            self.state = 'stuck'

    def progress_text(self):
        sc = self.sc
        if hasattr(sc, 'laps'):
            return f"Vuelta {min(int(sc.laps[0]) + 1, sc.required_laps)}/{sc.required_laps}"
        left = max(0, sc.start_dist - sc.progress[0])
        return f"Le faltan {left:.0f} px"

    def draw(self, surf, view, color, sensors=True, label=None, font=None):
        if len(self.trail) > 1:
            pygame.draw.lines(surf, color, False, [view.to_screen(p) for p in self.trail], 2)
        pos = view.to_screen(self.pos)
        if sensors and self.brain is not None and self.state == 'run':
            w = self.world
            for ang, dist in zip(w.sensor_angles, w.sensors[0]):
                a = self.angle + float(ang)
                end = view.to_screen((self.pos[0] + math.cos(a) * float(dist), self.pos[1] + math.sin(a) * float(dist)))
                pygame.draw.line(surf, (255, 120, 120), pos, end, 1)
                pygame.draw.circle(surf, (255, 70, 70), (int(end[0]), int(end[1])), 3)
        draw_car(surf, pos, self.angle, max(view.scale, 0.8), color)
        if self.state == 'crash':
            r = int(16 * max(view.scale, 0.6))
            pygame.draw.line(surf, ui.RED, (pos[0] - r, pos[1] - r), (pos[0] + r, pos[1] + r), 4)
            pygame.draw.line(surf, ui.RED, (pos[0] - r, pos[1] + r), (pos[0] + r, pos[1] - r), 4)
        if label and font:
            t = font.render(label, True, (255, 255, 255))
            box = t.get_rect(midbottom=(pos[0], pos[1] - 22 * max(view.scale, 0.6)))
            pygame.draw.rect(surf, (20, 22, 30), box.inflate(8, 4), border_radius=4)
            surf.blit(t, box)


def _new_maze_if_needed(scenario, meta):
    """En laberintos entrenados con laberintos que cambian, cada ronda usa uno nuevo"""
    maze = meta.get('maze') or {}
    if scenario.key == 'laberinto' and maze.get('new_every'):
        scenario.generate()
        return True
    return False


def _banner(surf, font, msg, center, color=(20, 22, 30)):
    t = font.render(msg, True, (255, 255, 255))
    r = t.get_rect(center=center)
    pygame.draw.rect(surf, color, r.inflate(30, 16), border_radius=10)
    surf.blit(t, r)


SIM_SPEEDS = [1, 2, 4]


# ---------------------------------------------------------------------- #
# Modo Expo
# ---------------------------------------------------------------------- #
class ExpoScreen:
    def __init__(self, screen, scenario, entry):
        self.screen = screen
        self.fonts = ui.Fonts()
        self.scenario = scenario
        self.entry = entry
        self.meta = entry['meta']
        self.cfg = entry['cfg']
        self.champ_brain = single_brain(entry['brain'])
        self.novice_brain, self.novice_label = novice_brain(entry)
        self.gen = self.meta.get('generation', '?')
        self.speed = 0
        self.paused = False
        if scenario.key == 'futbol':
            self.setup_football()
        else:
            self.setup_runs()

    # -------------------------------------------------------------- #
    def layout(self):
        W, H = self.screen.get_size()
        self.top, self.bottom = 70, H - 120
        return W, H

    def setup_runs(self):
        W, H = self.layout()
        half = (W - 45) // 2
        self.views = [MapView(self.scenario.background, pygame.Rect(15, self.top, half, self.bottom - self.top)),
                      MapView(self.scenario.background, pygame.Rect(30 + half, self.top, half, self.bottom - self.top))]
        self.novice = SoloRun(self.scenario, self.cfg, self.novice_brain)
        self.champ = SoloRun(self.scenario, self.cfg, self.champ_brain)
        self.attempts = 1
        self.wait = [0, 0]           # cuadros de espera antes de reiniciar cada lado
        self.best_time = None

    def setup_football(self):
        W, H = self.layout()
        sc = self.scenario
        self.view = MapView(sc.field.background, pygame.Rect(15, self.top, W - 30, self.bottom - self.top))
        self.sim = FootballSim(sc.field, self.cfg, 1, sc.team_size, True, control=sc.ball_control)
        self.match_steps = max(sc.match_steps, 1800)

    # -------------------------------------------------------------- #
    def step_runs(self):
        for k, run in enumerate((self.novice, self.champ)):
            if run.state == 'run':
                run.step()
            elif self.wait[k] == 0:
                self.wait[k] = 90 if k == 0 else 150
                if k == 1 and run.state == 'done':
                    t = run.step_n / 60
                    self.best_time = t if self.best_time is None else min(self.best_time, t)
            else:
                self.wait[k] -= 1
                if self.wait[k] == 0:
                    if k == 1:
                        # Ronda nueva: los dos arrancan de nuevo (laberinto nuevo si corresponde)
                        if _new_maze_if_needed(self.scenario, self.meta):
                            self.setup_runs()
                            return
                        self.champ.reset()
                        self.novice.reset()
                        self.attempts += 1
                        self.wait = [0, 0]
                    else:
                        self.novice.reset()
                        self.attempts += 1

    def step_football(self):
        sim = self.sim
        cars = np.arange(sim.C)
        team0 = cars[sim.team_of == 0]
        team1 = cars[sim.team_of == 1]
        steer = np.zeros(sim.C, np.float32)
        throttle, brake, kick = steer.copy(), steer.copy(), steer.copy()
        for brain, group in ((self.novice_brain, team0), (self.champ_brain, team1)):
            act = brain.forward(sim.observe(group), np.zeros(len(group), dtype=np.int64))
            steer[group], throttle[group], brake[group], kick[group] = sim.action_controls(act)
        sim.step(steer, throttle, brake, kick)
        if sim.steps >= self.match_steps:
            sim.reset_all()

    # -------------------------------------------------------------- #
    def draw_header(self, W):
        f = self.fonts
        ui.text(self.screen, "¿QUÉ APRENDIÓ LA IA?", f.big, ui.YELLOW, topleft=(20, 16))
        ui.text(self.screen, f"{self.entry['name']}", f.small, ui.TEXT_DIM, topleft=(22, 48))
        ui.text(self.screen, "Esc: salir  ·  Espacio: pausa  ·  1/2/3: velocidad", f.tiny, ui.TEXT_DIM,
                topright=(W - 20, 22))

    def draw_runs(self, W, H):
        f, scr = self.fonts, self.screen
        champ = tuple(self.cfg.color)
        sides = [(self.novice, faded(champ), self.novice_label.upper()),
                 (self.champ, champ, f"GENERACIÓN {self.gen}")]
        for k, (run, color, title) in enumerate(sides):
            view = self.views[k]
            scr.blit(view.bg, view.rect)
            self.scenario.draw_overlay(scr, view.to_screen, view.scale, f.tiny)
            run.draw(scr, view, color)
            pygame.draw.rect(scr, color, view.rect.inflate(6, 6), 3, border_radius=6)
            _banner(scr, f.normal, title, (view.rect.centerx, view.rect.y + 22), (20, 22, 30))
            if k == 0:
                status = {'crash': "¡Chocó!", 'stuck': "Se quedó dando vueltas", 'done': "¡Llegó!"}.get(run.state)
                info = f"Intento {self.attempts}  ·  {run.progress_text()}"
            else:
                status = {'crash': "Chocó", 'stuck': "Se trabó",
                          'done': f"¡Llegó en {run.step_n / 60:.1f} s!"}.get(run.state)
                best = f"  ·  Mejor tiempo: {self.best_time:.1f} s" if self.best_time else ""
                info = f"{run.progress_text()}{best}"
            ui.text(scr, info, f.small, ui.WHITE, midtop=(view.rect.centerx, view.rect.bottom + 8))
            if status:
                _banner(scr, f.big, status, view.rect.center, (150, 40, 40) if k == 0 and run.state != 'done' else (30, 120, 60))
        caption = ("Los dos autos tienen el mismo cuerpo, los mismos sensores y la misma red. "
                   f"Lo único distinto son los pesos: el de la derecha evolucionó {self.gen} generaciones.")
        self._caption(caption, W, H)

    def draw_football(self, W, H):
        f, scr, sim = self.fonts, self.screen, self.sim
        scr.blit(self.view.bg, self.view.rect)
        labels = None
        if sim.N <= 2:
            labels = {c: ("Gen 1" if sim.team_of[c] == 0 else f"Gen {self.gen}") for c in range(sim.C)}
        champ = tuple(self.cfg.color)
        draw_match(scr, sim, 0, self.view.to_screen, self.view.scale, labels, stripes=(faded(champ), champ))
        draw_score(scr, f.big, (self.view.rect.centerx, self.view.rect.y + 24), sim.score[0],
                   ("GEN 1", f"GEN {self.gen}"))
        sec = sim.steps // 60
        ui.text(scr, f"{sec // 60}:{sec % 60:02d}", f.normal, ui.WHITE, midtop=(self.view.rect.centerx, self.view.rect.y + 48))
        self._caption("Azules: cerebros de la generación 1. Rojos: el campeón después de "
                      f"{self.gen} generaciones. Mismo auto, misma red, distintos pesos.", W, H)

    def _caption(self, txt, W, H):
        from ai.menus import _wrap
        _wrap(self.screen, txt, self.fonts.small, ui.TEXT_DIM, pygame.Rect(40, H - 80, W - 80, 70), 22)

    # -------------------------------------------------------------- #
    def run(self):
        clock = pygame.time.Clock()
        football = self.scenario.key == 'futbol'
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'back'
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if pygame.K_1 <= event.key <= pygame.K_3:
                        self.speed = event.key - pygame.K_1
            if not self.paused:
                for _ in range(SIM_SPEEDS[self.speed]):
                    if football:
                        self.step_football()
                    else:
                        self.step_runs()
            W, H = self.screen.get_size()
            self.screen.fill(ui.BG)
            self.draw_header(W)
            if football:
                self.draw_football(W, H)
            else:
                self.draw_runs(W, H)
            if self.paused:
                _banner(self.screen, self.fonts.title, "PAUSA", (W // 2, H // 2))
            pygame.display.flip()
            clock.tick(60)


# ---------------------------------------------------------------------- #
# Competí contra la IA
# ---------------------------------------------------------------------- #
class RaceScreen:
    COUNTDOWN = 180

    def __init__(self, screen, scenario, entry):
        self.screen = screen
        self.fonts = ui.Fonts()
        self.scenario = scenario
        self.entry = entry
        self.meta = entry['meta']
        self.cfg = entry['cfg']
        self.brain = single_brain(entry['brain'])
        self.wins = [0, 0]
        self.reset()

    def reset(self):
        W, H = self.screen.get_size()
        area = pygame.Rect(15, 70, W - 30, H - 120)
        sc = self.scenario
        self.football = sc.key == 'futbol'
        self.countdown = self.COUNTDOWN
        self.result = None
        self.result_timer = 0
        if self.football:
            self.view = MapView(sc.field.background, area)
            self.sim = FootballSim(sc.field, self.cfg, 1, sc.team_size, True, control=sc.ball_control)
            self.player_car = 0
            self.match_steps = 3600
        else:
            self.view = MapView(sc.background, area)
            self.player = SoloRun(sc, self.cfg, None, bounce=True)
            self.ai = SoloRun(sc, self.cfg, self.brain, bounce=True)

    # -------------------------------------------------------------- #
    def step(self):
        if self.countdown > 0:
            self.countdown -= 1
            return
        if self.result:
            return
        controls = read_keys()
        if self.football:
            self.step_football(controls)
            return
        self.player.step(controls)
        self.ai.step()
        if self.ai.state == 'stuck':
            self.ai.state = 'run'     # en carrera la IA no queda afuera
            self.ai.sc.last_improve[0] = self.ai.step_n
        for k, run in enumerate((self.player, self.ai)):
            if run.state == 'done' and not self.result:
                self.result = 'player' if k == 0 else 'ai'
                self.wins[k] += 1

    def step_football(self, controls):
        sim = self.sim
        cars = np.arange(sim.C)
        steer = np.zeros(sim.C, np.float32)
        throttle, brake, kick = steer.copy(), steer.copy(), steer.copy()
        ai_cars = cars[cars != self.player_car]
        act = self.brain.forward(sim.observe(ai_cars), np.zeros(len(ai_cars), dtype=np.int64))
        steer[ai_cars], throttle[ai_cars], brake[ai_cars], kick[ai_cars] = sim.action_controls(act)
        p = self.player_car
        steer[p], throttle[p], brake[p] = controls
        kick[p] = 1.0 if pygame.key.get_pressed()[pygame.K_SPACE] else 0.0
        sim.step(steer, throttle, brake, kick)
        if sim.steps >= self.match_steps:
            a, b = int(sim.score[0, 0]), int(sim.score[0, 1])
            self.result = 'player' if a > b else ('ai' if b > a else 'draw')
            if self.result != 'draw':
                self.wins[0 if self.result == 'player' else 1] += 1

    # -------------------------------------------------------------- #
    def draw(self):
        f, scr = self.fonts, self.screen
        W, H = scr.get_size()
        scr.fill(ui.BG)
        ui.text(scr, "JUGÁ CONTRA LA IA", f.big, ui.YELLOW, topleft=(20, 16))
        keys = "Flechas: manejar" + ("  ·  Espacio: patear" if self.football else "")
        ui.text(scr, f"{keys}  ·  R: revancha  ·  Esc: salir", f.small, ui.TEXT_DIM, topleft=(22, 48))
        ui.text(scr, f"Vos {self.wins[0]}  -  {self.wins[1]} IA", f.big, ui.WHITE, topright=(W - 20, 18))
        v = self.view
        scr.blit(v.bg, v.rect)
        if self.football:
            sim = self.sim
            labels = {c: ("VOS" if c == self.player_car else "IA") for c in range(sim.C) if sim.N <= 2 or c == self.player_car}
            stripes = [None, None]
            stripes[int(sim.team_of[self.player_car])] = (255, 255, 255)
            stripes[1 - int(sim.team_of[self.player_car])] = tuple(self.cfg.color)
            draw_match(scr, sim, 0, v.to_screen, v.scale, labels, highlight=self.player_car, stripes=stripes)
            draw_score(scr, f.big, (v.rect.centerx, v.rect.y + 24), sim.score[0], ("VOS", "IA"))
            left = max(0, self.match_steps - sim.steps) // 60
            ui.text(scr, f"{left // 60}:{left % 60:02d}", f.normal, ui.WHITE, midtop=(v.rect.centerx, v.rect.y + 48))
            if sim.cooldown[self.player_car] == 0:
                ui.text(scr, "Patada lista", f.small, ui.GREEN, bottomleft=(v.rect.x + 10, v.rect.bottom - 8))
        else:
            self.scenario.draw_overlay(scr, v.to_screen, v.scale, f.tiny)
            self.ai.draw(scr, v, tuple(self.cfg.color), sensors=False, label="IA", font=f.small)
            self.player.draw(scr, v, PLAYER_COLOR, label="VOS", font=f.small)
            ui.text(scr, f"Vos: {self.player.progress_text()}   ·   IA: {self.ai.progress_text()}   ·   "
                    f"{self.player.step_n / 60:.1f} s", f.normal, ui.WHITE, midtop=(W // 2, v.rect.bottom + 10))
        if self.countdown > 0:
            n = self.countdown // 60 + 1
            _banner(scr, f.title, str(n), v.rect.center)
        elif self.countdown == 0 and self.result is None and (self.football and self.sim.steps < 60 or
                                                               not self.football and self.player.step_n < 60):
            _banner(scr, f.title, "¡YA!", v.rect.center, (30, 120, 60))
        if self.result:
            msg = {'player': "¡GANASTE!", 'ai': "Ganó la IA", 'draw': "Empate"}[self.result]
            _banner(scr, f.title, msg, v.rect.center, (30, 120, 60) if self.result == 'player' else (150, 40, 40))
            ui.text(scr, "R: revancha  ·  Esc: salir", f.normal, ui.WHITE, midtop=(v.rect.centerx, v.rect.centery + 40))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'back'
                    if event.key == pygame.K_r:
                        if not self.football and _new_maze_if_needed(self.scenario, self.meta):
                            pass
                        self.reset()
            self.step()
            self.draw()
            clock.tick(60)
