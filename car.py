import math
import pygame
import numpy as np  # Para normalización en observación (si se usa)

class Car:
    """
    Clase que representa el auto en el entorno.
    Maneja posición, rotación, movimiento y sensores de raycasting.
    Los sensores proyectan rayos en ángulos relativos para detectar distancias a bordes.
    """
    def __init__(self, environment):
        self.env = environment  # Referencia al entorno para acceder al track
        self.width_car = 20  # Ancho del auto (renombrado para evitar conflicto con env.width)
        self.height_car = 40  # Alto del auto
        # CORREGIDO: Posición inicial segura en la carretera blanca (parte inferior del óvalo)
        self.x = self.env.width / 2  # Centro horizontal
        self.y = self.env.height * 0.77  # 65% hacia abajo (dentro de la carretera blanca)
        self.angle = 0.0  # Ángulo en radianes (0 = hacia derecha)
        self.speed = 0.0  # Velocidad actual
        self.max_speed = 5.0  # Velocidad máxima por frame
        self.acceleration = 0.2  # Aceleración por frame
        self.friction = 0.05  # Fricción para desacelerar
        self.turn_speed = 0.1  # Velocidad de giro en radianes por frame
        
        # Ángulos relativos de los sensores (5 rayos: -60°, -30°, 0°, 30°, 60°)
        self.sensor_angles = [
            -math.pi / 3,  # -60 grados
            -math.pi / 6,  # -30 grados
            0,
            math.pi / 6,   # 30 grados
            math.pi / 3    # 60 grados
        ]
        # Distancia máxima de sensor: 20% de la menor dimensión, como entero
        self.max_sensor_distance = int(min(self.env.width, self.env.height) * 0.2)
        self.sensor_distances = [0] * len(self.sensor_angles)  # Almacena distancias para render

    def reset(self):
        """Reinicia la posición, ángulo y velocidad del auto (escalado a resolución)."""
        # CORREGIDO: Posición inicial segura en la carretera blanca
        self.x = self.env.width / 2  # Centro horizontal
        self.y = self.env.height * 0.77  # 65% hacia abajo (dentro de la carretera blanca)
        self.angle = 0.0  # Hacia la derecha
        self.speed = 0.0
        self.sensor_distances = [0] * len(self.sensor_angles)

    def apply_action(self, action):
        """
        Aplica una acción al auto.
        action: tupla (steering: -1 izquierda/0 ninguno/1 derecha, throttle: 0 no/1 sí)
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
        """Actualiza la posición y velocidad del auto (movimiento y fricción)."""
        # Aplicar fricción para desacelerar gradualmente
        if self.speed > 0:
            self.speed = max(self.speed - self.friction, 0)
        
        # Mover el auto basado en ángulo y velocidad
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        
        # CORREGIDO: Normalizar ángulo a [-π, π) para mejor manejo
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

    def get_sensor_distances(self, track):
        """
        Calcula las distancias de los sensores mediante raycasting.
        Devuelve lista de 5 floats (distancias).
        """
        distances = []
        for rel_angle in self.sensor_angles:
            dist = self._cast_ray(track, rel_angle)
            distances.append(dist)
        self.sensor_distances = distances  # Almacenar para render
        return distances

    def _cast_ray(self, track, rel_angle):
        """
        Proyecta un rayo desde la posición del auto en la dirección (angle + rel_angle).
        Avanza píxel por píxel hasta chocar con un borde negro o límite.
        OPTIMIZADO: Usa pasos más grandes y refina al detectar colisión.
        """
        direction_angle = self.angle + rel_angle
        dx = math.cos(direction_angle)
        dy = math.sin(direction_angle)
        
        x, y = self.x, self.y
        max_dist = self.max_sensor_distance  # Ya es int, no necesita conversión
        
        # Búsqueda gruesa (pasos de 5 píxeles para optimización)
        step_size = 5
        for distance in range(step_size, max_dist + 1, step_size):
            px = int(x + distance * dx)
            py = int(y + distance * dy)
            
            # Verificar límites de la pantalla
            if (px < 0 or px >= track.get_width() or
                py < 0 or py >= track.get_height()):
                # Refinar búsqueda en los últimos píxeles
                for fine_dist in range(max(1, distance - step_size), distance + 1):
                    px_fine = int(x + fine_dist * dx)
                    py_fine = int(y + fine_dist * dy)
                    if (px_fine < 0 or px_fine >= track.get_width() or
                        py_fine < 0 or py_fine >= track.get_height()):
                        return fine_dist
                return distance
            
            # Verificar si el píxel es un borde negro (colisión)
            if track.get_at((px, py)) == (0, 0, 0):
                # Refinar búsqueda en los últimos píxeles
                for fine_dist in range(max(1, distance - step_size), distance + 1):
                    px_fine = int(x + fine_dist * dx)
                    py_fine = int(y + fine_dist * dy)
                    if (px_fine >= 0 and px_fine < track.get_width() and
                        py_fine >= 0 and py_fine < track.get_height()):
                        if track.get_at((px_fine, py_fine)) == (0, 0, 0):
                            return fine_dist
                return distance
        
        return max_dist

    def draw(self, screen):
        """
        Dibuja el auto como un rectángulo rojo rotado en la pantalla.
        """
        # Crear superficie del auto (rectángulo simple)
        car_surf = pygame.Surface((self.width_car, self.height_car))
        car_surf.fill((255, 0, 0))  # Rojo
        
        # Rotar la superficie (PyGame usa grados, negativo para dirección correcta)
        rotated_surf = pygame.transform.rotate(car_surf, -math.degrees(self.angle))
        rotated_rect = rotated_surf.get_rect(center=(int(self.x), int(self.y)))
        
        screen.blit(rotated_surf, rotated_rect.topleft)

    def draw_sensors(self, screen):
        """
        Dibuja los sensores como líneas con color basado en distancia.
        MEJORADO: Color rojo=cerca, verde=lejos para mejor visualización.
        """
        for i, (rel_angle, dist) in enumerate(zip(self.sensor_angles, self.sensor_distances)):
            direction_angle = self.angle + rel_angle
            end_x = self.x + dist * math.cos(direction_angle)
            end_y = self.y + dist * math.sin(direction_angle)
            
            # Color basado en distancia (rojo si cerca, verde si lejos)
            ratio = dist / self.max_sensor_distance
            color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
            
            pygame.draw.line(screen, color, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)
