"""
Modo "Probar" del editor: manejás la pista con el teclado usando la misma física
y los mismos sensores que el auto de la IA (car.py), con las mismas reglas de
vueltas que el entrenamiento: hay que pasar todos los checkpoints y cruzar la
meta en el sentido de la flecha.
"""

import math
import pygame

from car import Car

SPEED_ZONE_COLOR = (0, 255, 0)
SLOW_ZONE_COLOR = (255, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 205, 40)
GREEN = (60, 200, 110)
RED = (230, 70, 70)
CYAN = (70, 210, 230)


class _TrackStub:
    """Lo mínimo que Car necesita de un Environment"""

    def __init__(self, width, height, spawn):
        self.width = width
        self.height = height
        self.custom_track_data = {'spawn_point': spawn}


def _segments_cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def format_time(ms):
    if ms is None:
        return "--"
    return f"{ms // 60000}:{(ms // 1000) % 60:02d}.{(ms % 1000) // 10:02d}"


class TestDrive:
    def __init__(self, road, speed, slow, spawn, finish, checkpoints, laps):
        self.road = road
        self.speed_layer = speed
        self.slow_layer = slow
        self.finish = finish
        self.checkpoints = checkpoints
        self.laps_needed = laps
        self.car = Car(_TrackStub(road.get_width(), road.get_height(), spawn))
        self.base_max_speed = self.car.max_speed
        self.best_lap = None
        self.restart()

    def restart(self):
        self.car.reset()
        self.passed = set()
        self.lap = 0
        self.start = pygame.time.get_ticks()
        self.lap_start = self.start
        self.crashed_at = None
        self.finished_at = None
        self.message = ("Flechas para manejar", WHITE)

    def zone(self):
        x, y = int(self.car.x), int(self.car.y)
        try:
            if self.speed_layer.get_at((x, y))[:3] == SPEED_ZONE_COLOR:
                return 'speed'
            if self.slow_layer.get_at((x, y))[:3] == SLOW_ZONE_COLOR:
                return 'slow'
        except IndexError:
            pass
        return None

    def update(self, keys):
        now = pygame.time.get_ticks()
        if self.crashed_at:
            if now - self.crashed_at > 1200:
                self.restart()
            return
        if self.finished_at:
            return

        steer = (-1 if keys[pygame.K_LEFT] else 0) + (1 if keys[pygame.K_RIGHT] else 0)
        self.car.max_speed = {'speed': 1.5, 'slow': 0.7}.get(self.zone(), 1.0) * self.base_max_speed
        prev = (self.car.x, self.car.y)
        self.car.apply_action((steer, 1 if keys[pygame.K_UP] else 0))
        if keys[pygame.K_DOWN]:
            self.car.speed = max(0.0, self.car.speed - 0.3)
        self.car.speed = min(self.car.speed, self.car.max_speed)
        self.car.update()
        cur = (self.car.x, self.car.y)
        self.car.get_sensor_distances(self.road)

        x, y = int(cur[0]), int(cur[1])
        w, h = self.road.get_size()
        if not (0 <= x < w and 0 <= y < h) or self.road.get_at((x, y))[:3] == (0, 0, 0):
            self.crashed_at = now
            self.message = ("¡Choque! Volvés a la salida...", RED)
            return

        for i, cp in enumerate(self.checkpoints):
            if i not in self.passed and _segments_cross(prev, cur, cp[:2], cp[2:4]):
                self.passed.add(i)
                self.message = (f"Checkpoint {i + 1}", CYAN)

        if self.finish and _segments_cross(prev, cur, self.finish[:2], self.finish[2:4]):
            fx1, fy1, fx2, fy2 = self.finish[:4]
            cross = (fx2 - fx1) * (cur[1] - prev[1]) - (fy2 - fy1) * (cur[0] - prev[0])
            if cross <= 0:
                self.message = ("Sentido contrario", RED)
            elif len(self.passed) < len(self.checkpoints):
                self.message = ("Faltan checkpoints: la vuelta no cuenta", YELLOW)
            else:
                lap_time = now - self.lap_start
                self.best_lap = lap_time if self.best_lap is None else min(self.best_lap, lap_time)
                self.lap += 1
                self.passed = set()
                self.lap_start = now
                if self.lap >= self.laps_needed:
                    self.finished_at = now
                    self.message = (f"¡Terminaste! Tiempo total {format_time(now - self.start)}", GREEN)
                else:
                    self.message = (f"Vuelta {self.lap} en {format_time(lap_time)}", GREEN)

    def draw(self, screen, to_screen, scale, font, small_font):
        car = self.car
        # Sensores: lo que "ve" la IA
        for rel, dist in zip(car.sensor_angles, car.sensor_distances):
            a = car.angle + rel
            end = (car.x + math.cos(a) * dist, car.y + math.sin(a) * dist)
            pygame.draw.line(screen, (255, 120, 120), to_screen((car.x, car.y)), to_screen(end), 1)
            pygame.draw.circle(screen, (255, 80, 80), [int(v) for v in to_screen(end)], 3)

        body = pygame.Surface((max(4, int(car.height_car * scale)), max(3, int(car.width_car * scale))),
                              pygame.SRCALPHA)
        body.fill(RED if self.crashed_at else CYAN)
        pygame.draw.rect(body, (20, 20, 30), (body.get_width() * 0.65, 1, body.get_width() * 0.25,
                                              body.get_height() - 2))
        rotated = pygame.transform.rotate(body, -math.degrees(car.angle))
        screen.blit(rotated, rotated.get_rect(center=to_screen((car.x, car.y))))

    def hud_lines(self):
        now = self.finished_at or pygame.time.get_ticks()
        return [
            f"Vuelta {min(self.lap + 1, self.laps_needed)}/{self.laps_needed}",
            f"Checkpoints {len(self.passed)}/{len(self.checkpoints)}",
            f"Tiempo {format_time(now - self.start)}",
            f"Mejor vuelta {format_time(self.best_lap)}",
            f"Velocidad {self.car.speed:.1f}",
        ]
