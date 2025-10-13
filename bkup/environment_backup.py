import pygame
import math
from car import Car
import numpy as np

class Environment:
    """
    Clase principal del entorno de simulación.
    Maneja la pista, el auto, colisiones, sensores y lógica de refuerzo.
    Preparado para Gym-like interface: reset(), step(action), render().
    La pista es un circuito ovalado simple (como NASCAR): carretera blanca en forma de óvalo
    con isla central negra y bordes exteriores negros.
    
    Cambios para DQN y resolución:
    - width/height parametrizados (default 1920x1080).
    - Pista escalada proporcionalmente para cualquier resolución.
    - No reset automático en step() (maneja done externamente).
    """
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.track = self._create_track()  # Surface de la pista con óvalo escalado
        self.car = Car(self)
        self.done = False
        self.action_space = 4  # Número de acciones discretas (para referencia)

    def _create_track(self):
        """
        Crea un circuito ovalado proceduralmente (estilo NASCAR simple), escalado a width/height.
        - Fondo negro (bordes exteriores y obstáculos).
        - Carretera: óvalo exterior blanco (10% márgenes).
        - Isla central: óvalo interior negro (ancho de carretera ~15-20% de height).
        """
        track = pygame.Surface((int(self.width), int(self.height)))  # Enteros explícitos
        track.fill((0, 0, 0))  # Fondo negro (obstáculos por defecto)
        
        # Márgenes proporcionales (10% de width/height)
        margin_x = int(self.width * 0.1)
        margin_y = int(self.height * 0.1)
        
        # Dibujar la carretera: óvalo exterior blanco (80% width, 70% height para forma ovalada)
        exterior_rect = pygame.Rect(margin_x, margin_y, int(self.width * 0.8), int(self.height * 0.7))
        pygame.draw.ellipse(track, (255, 255, 255), exterior_rect)
        
        # Dibujar isla central: óvalo interior negro (60% width, 50% height, centrado)
        inner_margin_x = int(self.width * 0.2)
        inner_margin_y = int(self.height * 0.25)
        interior_rect = pygame.Rect(inner_margin_x, inner_margin_y, int(self.width * 0.6), int(self.height * 0.5))
        pygame.draw.ellipse(track, (0, 0, 0), interior_rect)
        
        return track

    def reset(self):
        """
        Reinicia el entorno: posición del auto y estado.
        Devuelve la observación inicial normalizada (numpy array de 5 floats).
        """
        self.car.reset()
        self.done = False
        return self.car.get_sensor_distances(self.track)

    def step(self, action):
        """
        Ejecuta un paso en el entorno con acción discreta.
        action: int (0-3) o tupla (steering, throttle)
        Devuelve: (observación normalizada, recompensa, done, info)
        """
        # Si action es int, mapear a tupla (para compatibilidad con DQN)
        if isinstance(action, int):
            actions_map = [
                (0, 0),   # 0: nada
                (0, 1),   # 1: acel
                (-1, 1),  # 2: izquierda + acel
                (1, 1)    # 3: derecha + acel
            ]
            action = actions_map[action]
        
        # Aplicar acción y actualizar auto
        self.car.apply_action(action)
        self.car.update()
        
        # Calcular observación normalizada
        obs = self.car.get_sensor_distances(self.track)
        observation = np.array(obs) / self.car.max_sensor_distance
        
        # Verificar colisión
        collision = self._check_collision()
        
        if collision:
            reward = -10  # Penalización por choque
            self.done = True
        else:
            reward = 1  # Recompensa por avanzar sin chocar
            self.done = False
        
        info = {
            'position': (self.car.x, self.car.y),
            'angle': self.car.angle,
            'speed': self.car.speed
        }
        
        return observation, reward, self.done, info

    def _check_collision(self):
        """
        Verifica si el centro del auto está en un píxel de borde negro.
        """
        px, py = int(self.car.x), int(self.car.y)
        if (px < 0 or px >= self.width or py < 0 or py >= self.height):
            return True
        return self.track.get_at((px, py)) == (0, 0, 0)

    def render(self, screen):
        """
        Renderiza el entorno en la pantalla PyGame.
        - Blit pista.
        - Dibuja auto y sensores.
        screen: Surface de PyGame.
        """
        # Blit la pista de fondo
        screen.blit(self.track, (0, 0))
        
        # Dibujar auto y sensores
        self.car.draw(screen)
        self.car.draw_sensors(screen)