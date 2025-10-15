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
    def __init__(self, width=1920, height=1080, custom_track_data=None):
        self.width = width
        self.height = height
        
        # TASK 2 & 3: Custom track support
        self.custom_track_data = custom_track_data
        
        if custom_track_data:
            self.track = custom_track_data['track_layer']
            self.finish_line = custom_track_data.get('finish_line')
            self.checkpoints = custom_track_data.get('checkpoints', [])
            self.speed_zones = custom_track_data.get('speed_zones')
            self.slow_zones = custom_track_data.get('slow_zones')
            self.required_laps = custom_track_data.get('required_laps', 3)
        else:
            self.track = self._create_track()
            self.finish_line = None
            self.checkpoints = []
            self.speed_zones = None
            self.slow_zones = None
            self.required_laps = 1
        
        self.car = Car(self)
        self.done = False
        self.action_space = 4
        
        # Sistema de vueltas y checkpoints
        self.current_lap = 0
        self.checkpoints_passed = set()
        self.last_finish_cross = None
        self.finish_line_crossed = False

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
        CORREGIDO: Devuelve observación normalizada consistente con step().
        Devuelve solo sensores normalizados (5 valores) - gym_env agregará velocidad/ángulo.
        """
        self.car.reset()
        self.done = False
        
        # Reiniciar sistema de vueltas
        self.current_lap = 0
        self.checkpoints_passed = set()
        self.last_finish_cross = None
        self.finish_line_crossed = False
        
        obs = self.car.get_sensor_distances(self.track)
        # Normalizar distancias de sensores a [0, 1]
        observation = np.array(obs, dtype=np.float32) / self.car.max_sensor_distance
        return observation

    def step(self, action):
        """
        Ejecuta un paso en el entorno con acción discreta.
        action: int (0-3) o tupla (steering, throttle)
        CORREGIDO: Validación de acciones y recompensas mejoradas.
        Devuelve: (observación normalizada, recompensa, done, info)
        """
        # Guardar posición anterior para detectar cruces
        prev_pos = (self.car.x, self.car.y)
        
        # Validar acción
        if isinstance(action, int):
            if not 0 <= action < 4:
                raise ValueError(f"Acción inválida: {action}. Debe estar en [0, 3]")
            actions_map = [
                (0, 0),   # 0: nada
                (0, 1),   # 1: acel
                (-1, 1),  # 2: izquierda + acel
                (1, 1)    # 3: derecha + acel
            ]
            action = actions_map[action]
        elif isinstance(action, (list, tuple)):
            if len(action) != 2:
                raise ValueError(f"Acción debe tener 2 elementos, recibió {len(action)}")
        else:
            raise TypeError(f"Acción debe ser int o tupla, recibió {type(action)}")
        
        # Aplicar acción y actualizar auto
        self.car.apply_action(action)
        self.car.update()
        
        # Posición actual
        curr_pos = (self.car.x, self.car.y)
        
        # Calcular observación normalizada (solo sensores)
        obs = self.car.get_sensor_distances(self.track)
        observation = np.array(obs, dtype=np.float32) / self.car.max_sensor_distance
        
        # Verificar colisión
        collision = self._check_collision()
        
        # Variables para tracking
        lap_completed = False
        checkpoint_passed = False
        
        # MEJORADO: Sistema de recompensas más informativo
        if collision:
            reward = -100.0  # Penalización fuerte por choque
            self.done = True
        else:
            # Recompensa base proporcional a velocidad (fomenta ir rápido)
            reward = self.car.speed / self.car.max_speed
            
            # Verificar cruce de checkpoints
            if self.checkpoints:
                for i, checkpoint in enumerate(self.checkpoints):
                    if i not in self.checkpoints_passed:
                        if self._check_checkpoint_crossing(i, prev_pos, curr_pos):
                            self.checkpoints_passed.add(i)
                            reward += 10.0  # Bonus por checkpoint
                            checkpoint_passed = True
                            print(f"✓ Checkpoint {i+1}/{len(self.checkpoints)} pasado!")
            
            # Verificar cruce de línea de meta
            if self.finish_line:
                if self._check_finish_line_crossing(prev_pos, curr_pos):
                    # Solo cuenta si pasó todos los checkpoints
                    if len(self.checkpoints_passed) == len(self.checkpoints):
                        self.current_lap += 1
                        lap_completed = True
                        reward += 50.0  # Gran bonus por completar vuelta
                        print(f"✓ ¡Vuelta {self.current_lap} completada!")
                        
                        # Reiniciar checkpoints para próxima vuelta
                        self.checkpoints_passed = set()
                        
                        # Verificar si completó todas las vueltas requeridas
                        if self.current_lap >= self.required_laps:
                            reward += 100.0  # Bonus extra por completar todas las vueltas
                            self.done = True
                            print(f"✓ ¡Todas las vueltas completadas! ({self.required_laps})")
            
            # Penalizar si está muy cerca de bordes (sensor < 20% de max)
            min_sensor_dist = min(obs)
            if min_sensor_dist < self.car.max_sensor_distance * 0.2:
                reward -= 0.5  # Penalización por proximidad peligrosa
            
            # Bonus por mantener velocidad alta sin estar cerca de bordes
            if self.car.speed > self.car.max_speed * 0.7 and min_sensor_dist > self.car.max_sensor_distance * 0.3:
                reward += 0.5
            
            # Aplicar efecto de zonas especiales
            zone_effect = self._get_zone_effect(curr_pos)
            if zone_effect != 1.0:
                self.car.speed *= zone_effect
                self.car.speed = min(self.car.speed, self.car.max_speed)
        
        info = {
            'position': (self.car.x, self.car.y),
            'angle': self.car.angle,
            'speed': self.car.speed,
            'min_sensor': min(obs) if not collision else 0,
            'current_lap': self.current_lap,
            'checkpoints_passed': len(self.checkpoints_passed),
            'lap_completed': lap_completed,
            'checkpoint_passed': checkpoint_passed
        }
        
        return observation, reward, self.done, info

    def _check_collision(self):
        """
        Verifica si el centro del auto está en un píxel de borde negro.
        MEJORADO: Verifica múltiples puntos del auto para mejor detección.
        """
        # Verificar centro del auto
        px, py = int(self.car.x), int(self.car.y)
        if (px < 0 or px >= self.width or py < 0 or py >= self.height):
            return True
        
        if self.track.get_at((px, py)) == (0, 0, 0):
            return True
        
        # Verificar esquinas del auto (aproximación simple)
        half_width = self.car.width_car / 2
        half_height = self.car.height_car / 2
        
        # Calcular posiciones de esquinas rotadas
        cos_a = math.cos(self.car.angle)
        sin_a = math.sin(self.car.angle)
        
        corners = [
            (half_width, half_height),
            (-half_width, half_height),
            (half_width, -half_height),
            (-half_width, -half_height)
        ]
        
        for dx, dy in corners:
            # Rotar y trasladar
            rx = self.car.x + dx * cos_a - dy * sin_a
            ry = self.car.y + dx * sin_a + dy * cos_a
            
            px_corner = int(rx)
            py_corner = int(ry)
            
            # Verificar límites
            if (px_corner < 0 or px_corner >= self.width or 
                py_corner < 0 or py_corner >= self.height):
                return True
            
            # Verificar colisión
            if self.track.get_at((px_corner, py_corner)) == (0, 0, 0):
                return True
        
        return False

    def _check_line_crossing(self, line, prev_pos, curr_pos):
        """
        Verifica si el auto cruzó una línea entre dos posiciones.
        line: (x1, y1, x2, y2)
        Retorna: True si cruzó, False si no
        """
        if not line:
            return False
        
        x1, y1, x2, y2 = line[:4]
        px1, py1 = prev_pos
        px2, py2 = curr_pos
        
        # Algoritmo de intersección de segmentos
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        A = (x1, y1)
        B = (x2, y2)
        C = (px1, py1)
        D = (px2, py2)
        
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    def _check_finish_line_crossing(self, prev_pos, curr_pos):
        """Verifica si cruzó la línea de meta en la dirección correcta"""
        if not self.finish_line:
            return False
        
        if self._check_line_crossing(self.finish_line, prev_pos, curr_pos):
            # Verificar dirección (producto cruz)
            x1, y1, x2, y2, line_angle = self.finish_line
            
            # Vector de la línea
            line_dx = x2 - x1
            line_dy = y2 - y1
            
            # Vector de movimiento del auto
            move_dx = curr_pos[0] - prev_pos[0]
            move_dy = curr_pos[1] - prev_pos[1]
            
            # Producto cruz (positivo = dirección correcta)
            cross = line_dx * move_dy - line_dy * move_dx
            
            return cross > 0  # Solo cuenta si cruza en dirección correcta
        
        return False
    
    def _check_checkpoint_crossing(self, checkpoint_idx, prev_pos, curr_pos):
        """Verifica si cruzó un checkpoint específico"""
        if checkpoint_idx >= len(self.checkpoints):
            return False
        
        checkpoint = self.checkpoints[checkpoint_idx]
        return self._check_line_crossing(checkpoint, prev_pos, curr_pos)
    
    def _get_zone_effect(self, pos):
        """Obtiene el efecto de zona en la posición actual"""
        if not self.speed_zones and not self.slow_zones:
            return 1.0  # Sin efecto
        
        px, py = int(pos[0]), int(pos[1])
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return 1.0
        
        # Verificar speed zone (verde)
        if self.speed_zones:
            try:
                color = self.speed_zones.get_at((px, py))
                if color[:3] == (0, 255, 0):  # Verde
                    return 1.5  # 50% más rápido
            except:
                pass
        
        # Verificar slow zone (amarillo)
        if self.slow_zones:
            try:
                color = self.slow_zones.get_at((px, py))
                if color[:3] == (255, 255, 0):  # Amarillo
                    return 0.7  # 30% más lento
            except:
                pass
        
        return 1.0

    def render(self, screen):
        """
        Renderiza el entorno en la pantalla PyGame.
        - Blit pista.
        - Dibuja auto y sensores.
        screen: Surface de PyGame.
        """
        # Blit la pista de fondo
        screen.blit(self.track, (0, 0))
        
        # Dibujar capas especiales si existen
        if self.speed_zones:
            screen.blit(self.speed_zones, (0, 0))
        if self.slow_zones:
            screen.blit(self.slow_zones, (0, 0))
        
        # Dibujar checkpoints
        if self.checkpoints:
            for i, cp in enumerate(self.checkpoints):
                color = (0, 255, 0) if i in self.checkpoints_passed else (0, 100, 255)
                pygame.draw.line(screen, color, (cp[0], cp[1]), (cp[2], cp[3]), 8)
        
        # Dibujar línea de meta
        if self.finish_line:
            x1, y1, x2, y2 = self.finish_line[:4]
            pygame.draw.line(screen, (255, 0, 0), (x1, y1), (x2, y2), 10)
        
        # Dibujar auto y sensores
        self.car.draw(screen)
        self.car.draw_sensors(screen)
        
        # Mostrar info de vueltas si hay finish line
        if self.finish_line:
            font = pygame.font.Font(None, 28)
            lap_text = font.render(f"Vuelta: {self.current_lap}/{self.required_laps}", True, (255, 255, 255))
            pygame.draw.rect(screen, (0, 0, 0), (10, 50, 250, 35))
            screen.blit(lap_text, (15, 55))
