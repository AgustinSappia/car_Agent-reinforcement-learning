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
from ai.brain import PopulationBrain
from ai.config import brain_path, PROFILES_DIR
from ai.world import World

PANEL_W = 330
SPEEDS = [(1, "x1"), (2, "x2"), (5, "x5"), (15, "x15"), (40, "x40"), (0, "Turbo")]
TURBO_BUDGET = 0.035  # segundos de simulación por cuadro en modo turbo


class Trainer:
    def __init__(self, screen, scenario, config, profile='perfil', load_brain=False):
        self.screen = screen
        self.fonts = ui.Fonts()
        self.scenario = scenario
        self.cfg = config
        self.profile = profile
        self.clock = pygame.time.Clock()

        self.brain = PopulationBrain(config.layer_sizes(), config.population)
        self.loaded_note = ''
        path = brain_path(profile, scenario.key)
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
        self.world = World(sc.road_mask, self.cfg, sc.speed_mask, sc.slow_mask)
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
        self.step_count = 0
        self.reasons = {'choque': 0, 'llegó': 0, 'sin progreso': 0, 'tiempo': 0}

    def observe(self, idx):
        parts = [self.world.read_sensors(idx)]
        if self.cfg.use_speed:
            parts.append((self.world.speed[idx] / self.cfg.max_speed)[:, None])
        if self.cfg.use_compass:
            tx, ty = self.scenario.targets(idx)
            parts.append(self.world.compass(idx, tx, ty))
        return np.concatenate(parts, axis=1)

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
            self.reasons['llegó'] += int(done.sum())
            self.reasons['choque'] += int((crashed & ~done).sum())
            self.reasons['sin progreso'] += int((stalled & ~crashed & ~done).sum())
        self.step_count += 1
        self._steps_counter += len(idx)
        return True

    def end_generation(self):
        fit = self.scenario.fitness(self.cfg.max_steps)
        best_i = int(np.argmax(fit))
        reached = int((self.scenario.finished_step >= 0).sum())
        self.history.append((float(fit[best_i]), float(fit.mean()), reached))
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.generation, round(float(fit[best_i]), 1), round(float(fit.mean()), 1),
                                    reached, self.step_count])
        if fit[best_i] > self.best_ever:
            self.best_ever = float(fit[best_i])
            self.save_best(best_i)
        self.last_reason = dict(self.reasons)
        self.brain.evolve(fit, self.cfg.elite_pct, self.cfg.mutation_rate,
                          self.cfg.mutation_strength, self.cfg.crossover)
        self.generation += 1
        self.start_generation()

    def save_best(self, i):
        os.makedirs(PROFILES_DIR, exist_ok=True)
        self.brain.save_best(brain_path(self.profile, self.scenario.key), i,
                             {'fitness': self.best_ever, 'generation': self.generation})
        self.cfg.save(self.profile)

    def leader(self):
        alive = np.flatnonzero(self.world.alive)
        pool = alive if len(alive) else np.arange(self.cfg.population)
        return int(pool[np.argmax(self.scenario.progress[pool])])

    # ------------------------------------------------------------------ #
    # Interfaz
    # ------------------------------------------------------------------ #
    def panel_buttons(self):
        x, w = self.panel.x + 15, PANEL_W - 30
        y = self.screen.get_height() - 150
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
        return items

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
                    self.reasons['tiempo'] += 0
                elif kind == 'menu':
                    self.running = False

    def handle_key(self, event):
        k = event.key
        if k == pygame.K_ESCAPE:
            self.running = False
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
        s = self.scale
        L, Wd = 20 * s, 10 * s
        leader = self.leader()
        for i in alive:
            cx, cy = self.to_screen((w.x[i], w.y[i]))
            ca, sa = math.cos(float(w.angle[i])), math.sin(float(w.angle[i]))
            pts = [(cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)
                   for dx, dy in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd))]
            pygame.draw.polygon(self.screen, ui.AGENT_COLORS[i % len(ui.AGENT_COLORS)], pts)
        if w.alive[leader]:
            cx, cy = self.to_screen((w.x[leader], w.y[leader]))
            pygame.draw.circle(self.screen, ui.YELLOW, (int(cx), int(cy)), int(max(10, 28 * s)), 2)
            if self.show_sensors:
                for ang, dist in zip(w.sensor_angles, w.sensors[leader]):
                    a = w.angle[leader] + ang
                    end = self.to_screen((w.x[leader] + math.cos(a) * dist, w.y[leader] + math.sin(a) * dist))
                    pygame.draw.line(self.screen, (255, 110, 110), (cx, cy), end, 1)
                    pygame.draw.circle(self.screen, (255, 70, 70), (int(end[0]), int(end[1])), 3)
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

    def draw_panel(self, leader):
        f = self.fonts
        pygame.draw.rect(self.screen, ui.PANEL, self.panel)
        x = self.panel.x + 15
        y = 14
        ui.text(self.screen, f"{self.scenario.title.upper()}", f.big, ui.YELLOW, topleft=(x, y))
        ui.text(self.screen, ui.fit(self.scenario.goal_text, f.tiny, PANEL_W - 30), f.tiny, ui.TEXT_DIM, topleft=(x, y + 32))
        y += 56
        alive = int(self.world.alive.sum())
        rows = [
            ("Generación", str(self.generation)),
            ("Autos vivos", f"{alive}/{self.cfg.population}"),
            ("Paso", f"{self.step_count}/{self.cfg.max_steps}"),
            ("Mejor de todos", f"{self.best_ever:.0f}" if self.history else "-"),
            ("Velocidad sim.", f"{self.sim_rate:,.0f} pasos/s".replace(",", ".")),
        ]
        for label, value in rows:
            ui.text(self.screen, label, f.small, ui.TEXT_DIM, topleft=(x, y))
            ui.text(self.screen, value, f.small, ui.WHITE, topright=(self.panel.right - 15, y))
            y += 22
        for line in self.scenario.status():
            ui.text(self.screen, ui.fit(line, f.tiny, PANEL_W - 30), f.tiny, ui.TEXT_DIM, topleft=(x, y))
            y += 18
        if self.last_reason:
            r = self.last_reason
            line = (f"Gen. anterior: {r['llegó']} llegaron, {r['choque']} chocaron, "
                    f"{r['sin progreso'] + r['tiempo']} sin avanzar")
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
        ui.text(self.screen, f"Cerebro del líder  ·  {self.scenario.leader_text(leader)}", f.small, ui.WHITE, topleft=(x, y))
        y += 22
        net_h = self.screen.get_height() - 160 - y
        if net_h > 80:
            idx = np.array([leader])
            acts = None
            if self.world.alive[leader]:
                _, acts = self.brain.forward(self.observe(idx), idx, return_activations=True)
                acts = [a[0] for a in acts]
            ui.draw_network(self.screen, f, pygame.Rect(x, y, PANEL_W - 30, net_h), self.cfg.layer_sizes(), acts)

        for kind, k, label, rect in self.panel_buttons():
            active = kind == 'speed' and k == self.speed_idx
            color = ui.BLUE if active else ui.PANEL_LIGHT
            ui.button(self.screen, f.small, label, rect, color)

    def draw(self):
        self.screen.fill(ui.BG)
        self.screen.blit(self.bg_scaled, self.view)
        self.scenario.draw_overlay(self.screen, self.to_screen, self.scale, self.fonts.tiny)
        leader = self.draw_cars()
        self.draw_panel(leader)
        if self.paused:
            ui.text(self.screen, "PAUSA", self.fonts.title, ui.YELLOW, center=self.view.center)
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

            if not self.paused:
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
