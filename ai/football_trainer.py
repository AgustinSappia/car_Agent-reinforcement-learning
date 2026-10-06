"""
Entrenamiento del modo fútbol. Usa la misma pantalla que el entrenador de pistas,
pero cada generación es una ronda de partidos en paralelo.
"""

import csv
import math

import numpy as np
import pygame

from ai import ui
from ai.football import FootballSim, BOT_SPEED, draw_match, draw_score
from ai.trainer import Trainer, PANEL_W

LIVE_H = 116   # alto de la franja del puntaje en vivo, debajo de la cancha
LIVE_ITEMS = [
    ('goals', "Goles", lambda b: f"{b:.0f}"),
    ('conceded', "En contra", lambda b: f"{b:.0f}"),
    ('own_goals', "En propio arco", lambda b: f"{b:.0f}"),
    ('touches', "Toques", lambda b: f"{b:.0f}"),
    ('good_kicks', "Patadas útiles", lambda b: f"{b:.0f}"),
    ('advance', "Llevar al arco", lambda b: f"{b:.0f} px"),
    ('near', "Cerca de la pelota", lambda b: f"{b * 100:.0f} %"),
    ('crowd', "Amontonados", lambda b: f"{b * 100:.0f} %"),
    ('idle', "Quietos", lambda b: f"{b * 100:.0f} %"),
]


class FootballTrainer(Trainer):
    def _layout(self):
        """Como el de pistas, pero deja lugar abajo para el puntaje en vivo"""
        W, H = self.screen.get_size()
        mw, mh = self.scenario.size
        avail_w, avail_h = W - PANEL_W - 30, H - 30 - LIVE_H
        self.scale = min(avail_w / mw, avail_h / mh)
        vw, vh = int(mw * self.scale), int(mh * self.scale)
        self.view = pygame.Rect(15 + (avail_w - vw) // 2, 15 + (avail_h - vh) // 2, vw, vh)
        self.panel = pygame.Rect(W - PANEL_W, 0, PANEL_W, H)
        self.live = pygame.Rect(15, H - LIVE_H - 8, avail_w, LIVE_H)

    def _build_world(self):
        self._layout()
        self.bg_scaled = pygame.transform.smoothscale(self.scenario.field.background, self.view.size)
        self.scenario.changed = False
        self.rng = np.random.default_rng()

    def _init_log(self):
        super()._init_log()
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['generacion', 'mejor', 'promedio', 'goles_a_favor', 'goles_en_contra'])

    # ------------------------------------------------------------------ #
    def start_generation(self):
        sc, P = self.scenario, self.cfg.population
        two = sc.opponent != 'none'
        # Cada cerebro juega varios partidos y su puntaje es el promedio: un solo partido tiene mucha suerte
        K = max(1, self.cfg.eval_matches)
        if sc.opponent == 'self':
            # Partidos entre cerebros de la población, parejas al azar (otras parejas en cada ronda)
            rounds = []
            for _ in range(K):
                order = self.rng.permutation(P)
                if P % 2:
                    order = np.append(order, order[0])
                rounds.append(order.reshape(-1, 2))
            team_brain = np.concatenate(rounds)
        else:
            team_brain = np.tile(np.arange(P), K)[:, None]
            if two:
                team_brain = np.concatenate([team_brain, np.full((len(team_brain), 1), -1)], axis=1)
        self.team_brain = team_brain
        self.sim = FootballSim(sc.field, self.cfg, len(team_brain), sc.team_size, two, self.rng,
                               sc.ball_control, sc.random_starts)
        self.world = self.sim.world
        car_brain = team_brain[self.sim.match_of, self.sim.team_of]
        self.brain_cars = np.flatnonzero(car_brain >= 0)
        self.car_brain = car_brain
        self.bot_cars = np.flatnonzero(car_brain < 0)
        if len(self.bot_cars):
            self.world.speed_mult[self.bot_cars] = BOT_SPEED[sc.opponent]
        # Se muestra el partido del mejor de la generación anterior (queda primero tras la élite)
        self.featured = int(np.argwhere(team_brain == 0)[0][0])
        self.step_count = 0

    def sim_step(self):
        sim = self.sim
        if self.step_count >= self.scenario.match_steps:
            self.end_generation()
            return False
        C = sim.C
        steer = np.zeros(C, dtype=np.float32)
        throttle = np.zeros(C, dtype=np.float32)
        brake = np.zeros(C, dtype=np.float32)
        kick = np.zeros(C, dtype=np.float32)
        cars = self.brain_cars
        actions = self.brain.forward(sim.observe(cars), self.car_brain[cars])
        steer[cars], throttle[cars], brake[cars], kick[cars] = sim.action_controls(actions)
        if len(self.bot_cars):
            b = self.bot_cars
            steer[b], throttle[b], brake[b], kick[b] = sim.bot_controls(b)
        sim.step(steer, throttle, brake, kick)
        self.step_count += 1
        self._steps_counter += C
        return True

    def end_generation(self):
        sim, P = self.sim, self.cfg.population
        tf = sim.team_fitness()
        total = np.zeros(P, dtype=np.float64)
        count = np.zeros(P, dtype=np.int32)
        goals_for = goals_against = 0
        for t in range(self.team_brain.shape[1]):
            b = self.team_brain[:, t]
            ok = b >= 0
            np.add.at(total, b[ok], tf[ok, t])
            np.add.at(count, b[ok], 1)
        fit = (total / np.maximum(count, 1)).astype(np.float32)
        for m in range(len(self.team_brain)):
            for t in range(self.team_brain.shape[1]):
                if self.team_brain[m, t] >= 0:
                    goals_for += int(sim.score[m, t])
                    goals_against += int(sim.score[m, 1 - t])
        best_i = int(np.argmax(fit))
        self.history.append((float(fit[best_i]), float(fit.mean()), goals_for))
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.generation, round(float(fit[best_i]), 1), round(float(fit.mean()), 1),
                                    goals_for, goals_against])
        self.best_ever = max(self.best_ever, float(fit[best_i]))
        self.save_best(best_i, float(fit[best_i]))
        if self.generation == 1 and not self.loaded_note:
            # Para el modo Expo: un auto típico de la generación 1 (el del medio, no el mejor)
            typical = int(np.argsort(fit)[len(fit) // 2])
            self.save_best(typical, float(fit[typical]), suffix='_gen1')
        self.last_goals = (goals_for, goals_against, len(self.team_brain))
        self.brain.evolve(fit, self.cfg.elite_pct, self.cfg.mutation_rate,
                          self.cfg.mutation_strength, self.cfg.crossover)
        self.generation += 1
        self.start_generation()

    # ------------------------------------------------------------------ #
    def can_replay(self):
        return False

    def handle_key(self, event):
        if event.key == pygame.K_n:
            self.step_count = self.scenario.match_steps
            return
        super().handle_key(event)

    def handle_click(self, pos):
        for kind, _, _, rect in self.panel_buttons():
            if kind == 'skip' and rect.collidepoint(pos):
                self.step_count = self.scenario.match_steps
                return
        super().handle_click(pos)

    def leader(self):
        """Jugador del equipo azul del partido mostrado más cerca de la pelota"""
        sim, m = self.sim, self.featured
        cars = m * sim.PPM + np.arange(sim.N)
        d = np.hypot(sim.world.x[cars] - sim.bx[m], sim.world.y[cars] - sim.by[m])
        return int(cars[np.argmin(d)])

    def leader_text(self, leader):
        return f"Goles: {int(self.sim.goals_by[leader])}  ·  Toques: {int(self.sim.touches[leader])}"

    def leader_activations(self, leader):
        idx = np.array([leader])
        _, acts = self.brain.forward(self.sim.observe(idx), np.array([self.car_brain[leader]]),
                                     return_activations=True)
        return [a[0] for a in acts]

    def last_summary(self):
        if not getattr(self, 'last_goals', None):
            return ""
        gf, ga, m = self.last_goals
        return f"Gen. anterior: {gf} goles a favor, {ga} en contra ({m} partidos)"

    def panel_rows(self):
        sec = self.step_count // 60
        total = self.scenario.match_steps // 60
        return [
            ("Generación", str(self.generation)),
            ("Partidos a la vez", str(len(self.team_brain))),
            ("Tiempo", f"{sec // 60}:{sec % 60:02d} / {total // 60}:{total % 60:02d}"),
            ("Mejor de todos", f"{self.best_ever:.0f}" if self.history else "-"),
            ("Velocidad sim.", f"{self.sim_rate:,.0f} pasos/s".replace(",", ".")),
        ]

    def draw_cars(self):
        leader = self.leader()
        sim = self.sim
        if self.show_sensors:
            w = sim.world
            w.read_sensors(np.array([leader]))
            cx, cy = self.to_screen((w.x[leader], w.y[leader]))
            for ang, dist in zip(w.sensor_angles, w.sensors[leader]):
                a = float(w.angle[leader] + ang)
                end = self.to_screen((w.x[leader] + math.cos(a) * float(dist), w.y[leader] + math.sin(a) * float(dist)))
                pygame.draw.line(self.screen, (255, 230, 120), (cx, cy), end, 1)
        labels = None
        if len(self.bot_cars):
            labels = {c: "bot" for c in range(self.featured * sim.PPM, (self.featured + 1) * sim.PPM)
                      if self.car_brain[c] < 0}
        draw_match(self.screen, sim, self.featured, self.to_screen, self.scale, labels, highlight=leader,
                   stripes=(tuple(self.cfg.color), (40, 40, 40)))
        names = ("IA", "BOT" if len(self.bot_cars) else "IA")
        draw_score(self.screen, self.fonts.big, (self.view.centerx, self.view.y + 26), sim.score[self.featured], names)
        self.draw_live_score()
        return leader

    def draw_live_score(self):
        """Puntaje del equipo azul del partido que se ve, concepto por concepto, mientras se juega"""
        sim, f, scr, r = self.sim, self.fonts, self.screen, self.live
        parts = sim.team_parts(0)
        raw = sim.team_breakdown(0)
        brains = self.team_brain[:, 0] >= 0
        m = self.featured
        total = sum(float(v[m]) for v in parts.values())
        avg_total = sum(float(v[brains].mean()) for v in parts.values())
        pygame.draw.rect(scr, ui.PANEL, r, border_radius=10)
        ui.text(scr, f"PUNTAJE EN VIVO del equipo azul: {total:+.0f}", f.small, ui.YELLOW, topleft=(r.x + 12, r.y + 8))
        ui.text(scr, f"promedio de todos los partidos: {avg_total:+.0f}", f.tiny, ui.TEXT_DIM,
                topright=(r.right - 12, r.y + 10))
        biggest = max(1.0, max(abs(float(v[m])) for v in parts.values()))
        cols = 5
        cw = (r.w - 24) // cols
        for i, (key, label, fmt) in enumerate(LIVE_ITEMS):
            x = r.x + 12 + (i % cols) * cw
            y = r.y + 32 + (i // cols) * 42
            v = float(parts[key][m])
            avg = float(parts[key][brains].mean())
            ui.text(scr, ui.fit(f"{label}: {fmt(float(raw[key][m]))}", f.tiny, cw - 8), f.tiny, ui.TEXT_DIM, topleft=(x, y))
            color = ui.GREEN if v > 0 else (ui.RED if v < 0 else ui.TEXT_DIM)
            ui.text(scr, f"{v:+.0f}", f.small, color, topleft=(x, y + 15))
            ui.text(scr, f"prom. {avg:+.0f}", f.tiny, ui.TEXT_DIM, topright=(x + cw - 12, y + 18))
            bar = pygame.Rect(x, y + 34, int((cw - 14) * min(1.0, abs(v) / biggest)), 4)
            if bar.w:
                pygame.draw.rect(scr, color, bar, border_radius=2)
