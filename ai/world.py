"""
Simulación de todos los autos a la vez con numpy.

Misma física que car.py (aceleración 0.2, fricción 0.05), pero en vez de mover
los autos de a uno, se guardan posiciones, ángulos y velocidades en arrays y se
actualizan todos juntos. Los sensores también se calculan para todos los autos
en una sola operación.
"""

import math
import numpy as np
import pygame

ACCELERATION = 0.2
FRICTION = 0.05
BRAKE = 0.3
SENSOR_STEP = 4


def surface_mask(surface, color=None):
    """Matriz booleana [x, y]. Sin color: True donde no es negro. Con color: True donde coincide."""
    arr = pygame.surfarray.array3d(surface)
    if color is None:
        return np.any(arr != 0, axis=2)
    return np.all(arr == np.array(color, dtype=arr.dtype), axis=2)


class World:
    def __init__(self, road_mask, config, speed_mask=None, slow_mask=None):
        self.road = road_mask
        self.W, self.H = road_mask.shape
        self.speed_mask = speed_mask
        self.slow_mask = slow_mask
        self.cfg = config
        self.sensor_angles = np.array(config.sensor_angles(), dtype=np.float32)
        self.ray_steps = np.arange(SENSOR_STEP, config.sensor_range + 1, SENSOR_STEP, dtype=np.float32)
        self.actions = np.array(config.actions(), dtype=np.float32)  # (n_acciones, 3)

    def reset(self, n, x, y, angle):
        self.n = n
        self.x = np.full(n, x, dtype=np.float32)
        self.y = np.full(n, y, dtype=np.float32)
        self.angle = np.full(n, angle, dtype=np.float32)
        self.speed = np.zeros(n, dtype=np.float32)
        self.alive = np.ones(n, dtype=bool)
        self.sensors = np.zeros((n, len(self.sensor_angles)), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def on_road(self, x, y):
        xi = x.astype(np.int32)
        yi = y.astype(np.int32)
        inside = (xi >= 0) & (xi < self.W) & (yi >= 0) & (yi < self.H)
        result = np.zeros(x.shape, dtype=bool)
        result[inside] = self.road[xi[inside], yi[inside]]
        return result

    def read_sensors(self, idx):
        """Distancia a la pared de cada sensor, para los autos idx"""
        ang = self.angle[idx, None] + self.sensor_angles[None, :]               # (A, S)
        dx, dy = np.cos(ang), np.sin(ang)
        px = self.x[idx, None, None] + dx[..., None] * self.ray_steps          # (A, S, R)
        py = self.y[idx, None, None] + dy[..., None] * self.ray_steps
        hit = ~self.on_road(px, py)
        first = np.where(hit.any(axis=2), hit.argmax(axis=2), len(self.ray_steps))
        dist = first * SENSOR_STEP
        self.sensors[idx] = dist
        return dist / self.cfg.sensor_range

    def step(self, idx, action_idx):
        """Aplica las acciones elegidas y mueve los autos idx. Devuelve los que chocaron."""
        act = self.actions[action_idx]
        steer, throttle, brake = act[:, 0], act[:, 1], act[:, 2]

        max_speed = np.full(len(idx), self.cfg.max_speed, dtype=np.float32)
        if self.speed_mask is not None or self.slow_mask is not None:
            xi = np.clip(self.x[idx].astype(np.int32), 0, self.W - 1)
            yi = np.clip(self.y[idx].astype(np.int32), 0, self.H - 1)
            if self.speed_mask is not None:
                max_speed[self.speed_mask[xi, yi]] *= 1.5
            if self.slow_mask is not None:
                max_speed[self.slow_mask[xi, yi]] *= 0.7

        sp = self.speed[idx]
        sp = np.minimum(sp + ACCELERATION * throttle, max_speed)
        sp = np.maximum(sp - BRAKE * brake, 0)
        sp = np.maximum(sp - FRICTION, 0)
        ang = self.angle[idx] + steer * self.cfg.turn_speed
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        self.speed[idx] = sp
        self.angle[idx] = ang
        self.x[idx] += np.cos(ang) * sp
        self.y[idx] += np.sin(ang) * sp

        # Choque: centro o trompa del auto fuera del camino
        front_x = self.x[idx] + np.cos(ang) * 12
        front_y = self.y[idx] + np.sin(ang) * 12
        crashed = ~(self.on_road(self.x[idx], self.y[idx]) & self.on_road(front_x, front_y))
        return crashed

    def compass(self, idx, target_x, target_y):
        """Seno y coseno del ángulo hacia el objetivo, relativo al frente del auto"""
        rel = np.arctan2(target_y - self.y[idx], target_x - self.x[idx]) - self.angle[idx]
        return np.stack([np.sin(rel), np.cos(rel)], axis=1)
