import math
import pygame
import numpy as np

class Car:
    """
    Clase que representa el auto en el entorno.
    Maneja posición, rotación, movimiento y sensores de raycasting.
    Los sensores proyectan rayos en ángulos relativos para detectar distancias a bordes.
    """
    def __init__(self, environment):
        self.env = environment  # Referencia al entorno para acceder al track
        self.width_car = 20     # Ancho del auto
        self.height_car = 40    # Alto del auto

        # Posición inicial
        self.x = self.env.width / 2
        self.y = self.env.height * 0.77
        self.angle = -math.pi / 2  # Hacia arriba

        # Movimiento
        self.speed = 0.0
        self.max_speed = 5.0
        self.acceleration = 0.2
        self.friction = 0.05
        self.turn_speed = 0.1

        # Sensores (-60°, -30°, 0°, 30°, 60°)
        self.sensor_angles = [
            -math.pi / 3,
            -math.pi / 6,
            0,
            math.pi / 6,
            math.pi / 3
        ]
        self.max_sensor_distance = int(min(self.env.width, self.env.height) * 0.2)
        self.sensor_distances = [0] * len(self.sensor_angles)

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):
        """Reinicia la posición, ángulo y velocidad del auto."""
        if hasattr(self.env, 'custom_track_data') and self.env.custom_track_data:
            spawn_point = self.env.custom_track_data.get('spawn_point')
            if spawn_point and len(spawn_point) >= 3:
                self.x, self.y, self.angle = spawn_point[:3]
                self.speed = 0.0
                self.sensor_distances = [0] * len(self.sensor_angles)
                return
        
        # Por defecto
        self.x = self.env.width / 2
        self.y = self.env.height * 0.77
        self.angle = -math.pi / 2
        self.speed = 0.0
        self.sensor_distances = [0] * len(self.sensor_angles)

    # -----------------------------
    # ACCIONES Y MOVIMIENTO
    # -----------------------------
    def apply_action(self, action):
        """
        Aplica una acción al auto.
        action: tupla (steering: -1 izquierda / 0 ninguno / 1 derecha, throttle: 0 no / 1 sí)
        """
        steering, throttle = action
        if throttle > 0:
            self.accelerate()
        if steering < 0:
            self.turn_left()
        elif steering > 0:
            self.turn_right()

    def accelerate(self):
        """Aumenta la velocidad (con límite máximo)."""
        self.speed = min(self.speed + self.acceleration, self.max_speed)

    def turn_left(self):
        """Gira el auto a la izquierda."""
        self.angle -= self.turn_speed

    def turn_right(self):
        """Gira el auto a la derecha."""
        self.angle += self.turn_speed

    def update(self):
        """Actualiza la posición y velocidad del auto."""
        if self.speed > 0:
            self.speed = max(self.speed - self.friction, 0)

        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        # Mantener ángulo entre [-π, π)
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

    # -----------------------------
    # SENSORES
    # -----------------------------
    def get_sensor_distances(self, track):
        """
        Calcula las distancias de los sensores mediante raycasting.
        Devuelve lista de floats (distancias normalizadas).
        """
        distances = []
        for rel_angle in self.sensor_angles:
            dist = self._cast_ray(track, rel_angle)
            distances.append(dist)
        self.sensor_distances = distances
        return distances

    def _cast_ray(self, track, rel_angle):
        """
        Proyecta un rayo desde la posición del auto en la dirección (angle + rel_angle).
        Avanza píxel por píxel hasta chocar con un borde negro o límite.
        """
        direction_angle = self.angle + rel_angle
        dx = math.cos(direction_angle)
        dy = math.sin(direction_angle)

        x, y = self.x, self.y
        max_dist = self.max_sensor_distance
        step_size = 5

        for distance in range(step_size, max_dist + 1, step_size):
            px = int(x + distance * dx)
            py = int(y + distance * dy)

            if px < 0 or px >= track.get_width() or py < 0 or py >= track.get_height():
                return distance

            try:
                color = track.get_at((px, py))[:3]
            except Exception:
                color = (0, 0, 0)

            if color == (0, 0, 0):  # Borde o fuera de pista
                # Refinar en pasos pequeños
                for fine_dist in range(distance - step_size, distance + 1):
                    px_fine = int(x + fine_dist * dx)
                    py_fine = int(y + fine_dist * dy)
                    if 0 <= px_fine < track.get_width() and 0 <= py_fine < track.get_height():
                        try:
                            fine_color = track.get_at((px_fine, py_fine))[:3]
                        except Exception:
                            fine_color = (0, 0, 0)
                        if fine_color == (0, 0, 0):
                            return fine_dist
                return distance

        return max_dist

    # -----------------------------
    # RENDERIZADO
    # -----------------------------
    def draw(self, screen):
        """
        Dibuja el auto como un rectángulo rojo rotado.
        """
        car_surf = pygame.Surface((self.width_car, self.height_car), pygame.SRCALPHA)
        car_surf.fill((255, 0, 0, 220))  # Rojo con algo de transparencia

        rotated_surf = pygame.transform.rotate(car_surf, -math.degrees(self.angle))
        rotated_rect = rotated_surf.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated_surf, rotated_rect.topleft)

    def draw_sensors(self, screen):
        """
        Dibuja los sensores como líneas con color según distancia.
        Rojo = cerca, verde = lejos.
        """
        for rel_angle, dist in zip(self.sensor_angles, self.sensor_distances):
            direction_angle = self.angle + rel_angle
            end_x = self.x + dist * math.cos(direction_angle)
            end_y = self.y + dist * math.sin(direction_angle)

            ratio = dist / self.max_sensor_distance
            color = (int(255 * (1 - ratio)), int(255 * ratio), 0)

            pygame.draw.line(screen, color, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)
