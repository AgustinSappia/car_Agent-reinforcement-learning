"""
Pantalla de entrenamiento genético.

Cada generación todos los autos salen juntos. Cuando todos terminaron (chocaron,
llegaron o se quedaron sin progresar), los mejores pasan a la siguiente generación
y el resto se reemplaza por hijos mutados de los mejores.
"""

import csv
import math
import os
import time
from datetime import datetime

import numpy as np
import pygame

from ai import ui
from ai import hall
from ai.brain import PopulationBrain
from ai.sprites import draw_car, variant
from ai.config import brain_path, PROFILES_DIR
from ai.world import World, observe

PANEL_W = 330
SPEEDS = [(1, "x1"), (2, "x2"), (5, "x5"), (15, "x15"), (40, "x40"), (0, "Turbo")]
TURBO_BUDGET = 0.035  # segundos de simulación por cuadro en modo turbo
REPLAY_SPEED = 3      # pasos por cuadro en la repetición


class Trainer:
    def __init__(self, screen, scenario, config, profile='perfil', load_brain=False):
        self.screen = screen
        self.fonts = ui.Fonts()
        self.scenario = scenario
        self.cfg = config
        self.profile = profile
        self.clock = pygame.time.Clock()

        self.kind = scenario.key
        self.brain = PopulationBrain(config.layer_sizes(), config.population)
        self.loaded_note = ''
        path = brain_path(profile, self.kind)
        if load_brain and os.path.exists(path):
            ok = self.brain.load_into_all(path, mutation_strength=config.mutation_strength)
            self.loaded_note = "Continúa desde el cerebro guardado" if ok else "El cerebro guardado no es compatible: empieza de cero"

        self.generation = 1
        self.history = []          # (mejor, promedio, llegaron)
        self.best_ever = -1e9
        self.speed_idx = 2
        self.paused = False
        self.show_sensors = True
        self.running = True
        self.result = None
        self.last_reason = {}
        self.replay = None         # mejor recorrido de la última generación
        self.replaying = None      # [paso actual] mientras se muestra la repetición
        self.toast = ("", 0)
        self.sim_rate = 0.0
        self._steps_counter, self._rate_t = 0, time.time()

        self._layout()
        self._build_world()
        self._init_log()
        self.start_generation()

    # ------------------------------------------------------------------ #
    def _layout(self):
        W, H = self.screen.get_size()
        mw, mh = self.scenario.size
        avail_w, avail_h = W - PANEL_W - 30, H - 30
        self.scale = min(avail_w / mw, avail_h / mh)
        vw, vh = int(mw * self.scale), int(mh * self.scale)
        self.view = pygame.Rect(15 + (avail_w - vw) // 2, 15 + (avail_h - vh) // 2, vw, vh)
        self.panel = pygame.Rect(W - PANEL_W, 0, PANEL_W, H)

    def _build_world(self):
        sc = self.scenario
        self._layout()
        self.world = World(sc.road_mask, self.cfg, sc.speed_mask, sc.slow_mask, self.kind)
        self.bg_scaled = pygame.transform.smoothscale(sc.background, self.view.size)
        sc.changed = False

    def _init_log(self):
        os.makedirs('logs', exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join('logs', f"{self.profile}_{self.scenario.key}_{stamp}.csv")
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['generacion', 'mejor', 'promedio', 'completaron', 'pasos'])

    def to_screen(self, p):
        return (float(self.view.x + p[0] * self.scale), float(self.view.y + p[1] * self.scale))

    # ------------------------------------------------------------------ #
    # Simulación
    # ------------------------------------------------------------------ #
    def start_generation(self):
        P = self.cfg.population
        self.scenario.start_generation(P, self.generation)
        if self.scenario.changed:
            self._build_world()
        x, y, a = self.scenario.spawn
        self.world.reset(P, x, y, a)
        steps = self.cfg.max_steps + 1
        if getattr(self, 'rec', None) is None or self.rec.shape[:2] != (steps, P):
            self.rec = np.zeros((steps, P, 3), dtype=np.float32)
        self.rec[0] = np.stack([self.world.x, self.world.y, self.world.angle], axis=1)
        self.death_step = np.full(P, self.cfg.max_steps, dtype=np.int32)
        self.crashed = np.zeros(P, dtype=bool)
        self.step_count = 0
        self.reasons = {'choque': 0, 'llegó': 0, 'sin progreso': 0, 'tiempo': 0}

    def observe(self, idx):
        return observe(self.world, self.scenario, self.cfg, idx)

    def sim_step(self):
        """Avanza un paso a todos los autos vivos. Devuelve False si terminó la generación."""
        w, sc = self.world, self.scenario
        idx = np.flatnonzero(w.alive)
        if len(idx) == 0 or self.step_count >= self.cfg.max_steps:
            self.reasons['tiempo'] += len(idx)
            self.end_generation()
            return False
        obs = self.observe(idx)
        actions = self.brain.forward(obs, idx)
        prev_x, prev_y = w.x[idx].copy(), w.y[idx].copy()
        crashed = w.step(idx, actions)
        done = sc.update(w, idx, prev_x, prev_y, self.step_count)
        stalled = (self.step_count - sc.last_improve[idx]) > self.cfg.patience
        dead = crashed | done | stalled
        if dead.any():
            w.alive[idx[dead]] = False
            self.death_step[idx[dead]] = self.step_count + 1
            self.reasons['llegó'] += int(done.sum())
            self.reasons['choque'] += int((crashed & ~done).sum())
            self.crashed[idx[crashed & ~done]] = True
            self.reasons['sin progreso'] += int((stalled & ~crashed & ~done).sum())
        self.step_count += 1
        self.rec[self.step_count] = np.stack([w.x, w.y, w.angle], axis=1)
        self._steps_counter += len(idx)
        return True

    def end_generation(self):
        fit = self.scenario.fitness(self.cfg) - self.crashed * self.cfg.p_crash
        best_i = int(np.argmax(fit))
        reached = int((self.scenario.finished_step >= 0).sum())
        self.history.append((float(fit[best_i]), float(fit.mean()), reached))
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.generation, round(float(fit[best_i]), 1), round(float(fit.mean()), 1),
                                    reached, self.step_count])
        self.best_ever = max(self.best_ever, float(fit[best_i]))
        # Se guarda siempre el mejor de la generación: con élite nunca es peor que el anterior
        self.save_best(best_i, float(fit[best_i]))
        if self.generation == 1 and not self.loaded_note:
            # Para el modo Expo: un auto típico de la generación 1 (el del medio, no el mejor)
            typical = int(np.argsort(fit)[len(fit) // 2])
            self.save_best(typical, float(fit[typical]), suffix='_gen1')
        self.store_replay(best_i, float(fit[best_i]))
        self.last_reason = dict(self.reasons)
        self.brain.evolve(fit, self.cfg.elite_pct, self.cfg.mutation_rate,
                          self.cfg.mutation_strength, self.cfg.crossover)
        self.generation += 1
        self.start_generation()

    def meta(self, fitness):
        info = {'kind': self.kind, 'title': self.scenario.title, 'profile': self.profile,
                'generation': self.generation, 'fitness': round(fitness, 1),
                'date': datetime.now().strftime("%d/%m/%Y %H:%M"), 'config': self.cfg.to_dict()}
        info.update(self.scenario.map_info())
        return info

    def save_best(self, i, fitness, suffix=''):
        os.makedirs(os.path.join(PROFILES_DIR, self.kind), exist_ok=True)
        self.brain.save_best(brain_path(self.profile, self.kind, suffix), i, self.meta(fitness))
        if not suffix:
            self.cfg.save(self.profile)

    def save_champion(self):
        path = brain_path(self.profile, self.kind)
        if not os.path.exists(path):
            self.flash("Todavía no terminó ninguna generación")
            return
        name = hall.save_champion(path, brain_path(self.profile, self.kind, '_gen1'))
        self.flash(f"Guardado en la galería: {name}")

    def flash(self, msg):
        self.toast = (msg, pygame.time.get_ticks() + 3000)

    # ------------------------------------------------------------------ #
    # Repetición del mejor
    # ------------------------------------------------------------------ #
    def store_replay(self, i, fitness):
        end = min(int(self.death_step[i]), self.step_count)
        self.replay = {'gen': self.generation, 'path': self.rec[:end + 1, i].copy(), 'fitness': fitness,
                       'finished': bool(self.scenario.finished_step[i] >= 0)}

    def start_replay(self):
        if self.replay is None:
            self.flash("La repetición aparece al terminar la primera generación")
            return
        self.replaying = [0.0]

    def draw_replay(self):
        path = self.replay['path']
        k = min(int(self.replaying[0]), len(path) - 1)
        pts = [self.to_screen(p[:2]) for p in path[:k + 1:3]]
        if len(pts) > 1:
            pygame.draw.lines(self.screen, ui.YELLOW, False, pts, 3)
        x, y, a = path[k]
        cx, cy = self.to_screen((x, y))
        draw_car(self.screen, (cx, cy), float(a), self.scale, self.cfg.color)
        pygame.draw.circle(self.screen, ui.WHITE, (int(cx), int(cy)), int(max(12, 30 * self.scale)), 2)
        end = "llegó" if self.replay['finished'] else "no llegó"
        msg = f"REPETICIÓN: el mejor de la generación {self.replay['gen']} ({end})  ·  R o Esc para seguir"
        r = ui.text(self.screen, msg, self.fonts.small, ui.WHITE, midtop=(self.view.centerx, self.view.y + 10))
        pygame.draw.rect(self.screen, (20, 22, 30), r.inflate(20, 10), border_radius=8)
        ui.text(self.screen, msg, self.fonts.small, ui.WHITE, midtop=(self.view.centerx, self.view.y + 10))
        self.replaying[0] += REPLAY_SPEED
        if self.replaying[0] >= len(path) + 60:
            self.replaying = None

    def leader(self):
        alive = np.flatnonzero(self.world.alive)
        pool = alive if len(alive) else np.arange(self.cfg.population)
        return int(pool[np.argmax(self.scenario.progress[pool])])

    # ------------------------------------------------------------------ #
    # Interfaz
    # ------------------------------------------------------------------ #
    def panel_buttons(self):
        x, w = self.panel.x + 15, PANEL_W - 30
        y = self.screen.get_height() - 186
        bw = (w - 10) // 3
        items = []
        for k, (_, label) in enumerate(SPEEDS):
            items.append(('speed', k, label, pygame.Rect(x + (k % 3) * (bw + 5), y + (k // 3) * 36, bw, 30)))
        y += 76
        half = (w - 6) // 2
        items.append(('pause', 0, "Reanudar (Espacio)" if self.paused else "Pausa (Espacio)",
                      pygame.Rect(x, y, half, 30)))
        items.append(('sensors', 0, "Sensores: sí" if self.show_sensors else "Sensores: no",
                      pygame.Rect(x + half + 6, y, half, 30)))
        y += 36
        items.append(('skip', 0, "Siguiente gen. (N)", pygame.Rect(x, y, half, 30)))
        items.append(('menu', 0, "Menú (Esc)", pygame.Rect(x + half + 6, y, half, 30)))
        y += 36
        if self.can_replay():
            items.append(('replay', 0, "Ver al mejor (R)", pygame.Rect(x, y, half, 30)))
            items.append(('champion', 0, "A la galería (G)", pygame.Rect(x + half + 6, y, half, 30)))
        else:
            items.append(('champion', 0, "Guardar en la galería (G)", pygame.Rect(x, y, w, 30)))
        return items

    def can_replay(self):
        return True

    def handle_click(self, pos):
        for kind, k, _, rect in self.panel_buttons():
            if rect.collidepoint(pos):
                if kind == 'speed':
                    self.speed_idx = k
                elif kind == 'pause':
                    self.paused = not self.paused
                elif kind == 'sensors':
                    self.show_sensors = not self.show_sensors
                elif kind == 'skip':
                    self.world.alive[:] = False
                elif kind == 'menu':
                    self.running = False
                elif kind == 'replay':
                    self.start_replay()
                elif kind == 'champion':
                    self.save_champion()

    def handle_key(self, event):
        k = event.key
        if self.replaying is not None and k in (pygame.K_ESCAPE, pygame.K_r):
            self.replaying = None
        elif k == pygame.K_ESCAPE:
            self.running = False
        elif k == pygame.K_r and self.can_replay():
            self.start_replay()
        elif k == pygame.K_g:
            self.save_champion()
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_s:
            self.show_sensors = not self.show_sensors
        elif k == pygame.K_n:
            self.world.alive[:] = False
        elif pygame.K_1 <= k <= pygame.K_6:
            self.speed_idx = k - pygame.K_1

    # ------------------------------------------------------------------ #
    def draw_cars(self):
        w = self.world
        alive = np.flatnonzero(w.alive)
        leader = self.leader()
        for i in alive:
            if i != leader:
                draw_car(self.screen, self.to_screen((w.x[i], w.y[i])), float(w.angle[i]), self.scale,
                         variant(self.cfg.color, i))
        if w.alive[leader]:
            cx, cy = self.to_screen((w.x[leader], w.y[leader]))
            if self.show_sensors:
                for ang, dist in zip(w.sensor_angles, w.sensors[leader]):
                    a = w.angle[leader] + ang
                    end = self.to_screen((w.x[leader] + math.cos(a) * dist, w.y[leader] + math.sin(a) * dist))
                    pygame.draw.line(self.screen, (255, 110, 110), (cx, cy), end, 1)
                    pygame.draw.circle(self.screen, (255, 70, 70), (int(end[0]), int(end[1])), 3)
            draw_car(self.screen, (cx, cy), float(w.angle[leader]), self.scale, self.cfg.color)
            pygame.draw.circle(self.screen, ui.YELLOW, (int(cx), int(cy)), int(max(12, 30 * self.scale)), 2)
        return leader

    def draw_chart(self, rect):
        pygame.draw.rect(self.screen, (28, 30, 40), rect, border_radius=8)
        data = self.history[-60:]
        if len(data) < 2:
            ui.text(self.screen, "El gráfico aparece después", self.fonts.tiny, ui.TEXT_DIM, center=(rect.centerx, rect.centery - 8))
            ui.text(self.screen, "de 2 generaciones", self.fonts.tiny, ui.TEXT_DIM, center=(rect.centerx, rect.centery + 8))
            return
        best = [d[0] for d in data]
        avg = [d[1] for d in data]
        lo, hi = min(min(avg), 0), (max(best) or 1) * 1.15
        def pts(vals):
            return [(rect.x + 8 + (rect.w - 16) * i / (len(vals) - 1),
                     rect.bottom - 8 - (rect.h - 16) * (v - lo) / (hi - lo or 1)) for i, v in enumerate(vals)]
        pygame.draw.lines(self.screen, ui.TEXT_DIM, False, pts(avg), 2)
        pygame.draw.lines(self.screen, ui.GREEN, False, pts(best), 2)
        ui.text(self.screen, "mejor", self.fonts.tiny, ui.GREEN, topleft=(rect.x + 8, rect.y + 6))
        ui.text(self.screen, "promedio", self.fonts.tiny, ui.TEXT_DIM, topleft=(rect.x + 55, rect.y + 6))

    def last_summary(self):
        if not self.last_reason:
            return ""
        r = self.last_reason
        return (f"Gen. anterior: {r['llegó']} llegaron, {r['choque']} chocaron, "
                f"{r['sin progreso'] + r['tiempo']} sin avanzar")

    def leader_text(self, leader):
        return self.scenario.leader_text(leader)

    def leader_activations(self, leader):
        if not self.world.alive[leader]:
            return None
        idx = np.array([leader])
        _, acts = self.brain.forward(self.observe(idx), idx, return_activations=True)
        return [a[0] for a in acts]

    def panel_rows(self):
        alive = int(self.world.alive.sum())
        return [
            ("Generación", str(self.generation)),
            ("Autos vivos", f"{alive}/{self.cfg.population}"),
            ("Paso", f"{self.step_count}/{self.cfg.max_steps}"),
            ("Mejor de todos", f"{self.best_ever:.0f}" if self.history else "-"),
            ("Velocidad sim.", f"{self.sim_rate:,.0f} pasos/s".replace(",", ".")),
        ]

    def draw_panel(self, leader):
        f = self.fonts
        pygame.draw.rect(self.screen, ui.PANEL, self.panel)
        x = self.panel.x + 15
        y = 14
        ui.text(self.screen, f"{self.scenario.title.upper()}", f.big, ui.YELLOW, topleft=(x, y))
        ui.text(self.screen, ui.fit(self.scenario.goal_text, f.tiny, PANEL_W - 30), f.tiny, ui.TEXT_DIM, topleft=(x, y + 32))
        y += 56
        for label, value in self.panel_rows():
            ui.text(self.screen, label, f.small, ui.TEXT_DIM, topleft=(x, y))
            ui.text(self.screen, value, f.small, ui.WHITE, topright=(self.panel.right - 15, y))
            y += 22
        for line in self.scenario.status():
            ui.text(self.screen, ui.fit(line, f.tiny, PANEL_W - 30), f.tiny, ui.TEXT_DIM, topleft=(x, y))
            y += 18
        line = self.last_summary()
        if line:
            ui.text(self.screen, ui.fit(line, f.tiny, PANEL_W - 30), f.tiny, ui.TEXT_DIM, topleft=(x, y))
            y += 18
        if self.loaded_note:
            ui.text(self.screen, ui.fit(self.loaded_note, f.tiny, PANEL_W - 30), f.tiny, ui.CYAN, topleft=(x, y))
            y += 18

        y += 6
        chart_h = 110
        self.draw_chart(pygame.Rect(x, y, PANEL_W - 30, chart_h))
        y += chart_h + 10

        # Cerebro del líder
        ui.text(self.screen, f"Cerebro del líder  ·  {self.leader_text(leader)}", f.small, ui.WHITE, topleft=(x, y))
        y += 22
        net_h = self.screen.get_height() - 196 - y
        if net_h > 80:
            acts = self.leader_activations(leader)
            ui.draw_network(self.screen, f, pygame.Rect(x, y, PANEL_W - 30, net_h), self.cfg.layer_sizes(), acts)

        for kind, k, label, rect in self.panel_buttons():
            active = kind == 'speed' and k == self.speed_idx
            color = ui.BLUE if active else ui.PANEL_LIGHT
            ui.button(self.screen, f.small, label, rect, color)

    def draw(self):
        self.screen.fill(ui.BG)
        self.screen.blit(self.bg_scaled, self.view)
        self.scenario.draw_overlay(self.screen, self.to_screen, self.scale, self.fonts.tiny)
        if self.replaying is not None:
            self.draw_replay()
            leader = self.leader()
        else:
            leader = self.draw_cars()
        self.draw_panel(leader)
        if self.paused and self.replaying is None:
            ui.text(self.screen, "PAUSA", self.fonts.title, ui.YELLOW, center=self.view.center)
        msg, until = self.toast
        if msg and pygame.time.get_ticks() < until:
            r = ui.text(self.screen, msg, self.fonts.small, ui.WHITE, midbottom=(self.view.centerx, self.view.bottom - 10))
            pygame.draw.rect(self.screen, ui.PANEL_LIGHT, r.inflate(24, 12), border_radius=8)
            ui.text(self.screen, msg, self.fonts.small, ui.WHITE, midbottom=(self.view.centerx, self.view.bottom - 10))
        pygame.display.flip()

    # ------------------------------------------------------------------ #
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.result = 'quit'
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            if not self.paused and self.replaying is None:
                steps = SPEEDS[self.speed_idx][0]
                if steps:
                    for _ in range(steps):
                        self.sim_step()
                else:
                    t0 = time.time()
                    while time.time() - t0 < TURBO_BUDGET:
                        self.sim_step()

            now = time.time()
            if now - self._rate_t > 0.5:
                self.sim_rate = self._steps_counter / (now - self._rate_t)
                self._steps_counter, self._rate_t = 0, now
            self.draw()
            self.clock.tick(60)
        return self.result
