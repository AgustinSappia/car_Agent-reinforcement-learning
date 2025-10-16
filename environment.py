# environment.py
"""
Environment corregido y robusto para integrarse con:
- track_editor_v2.py (editor)
- train_genetic.py (trainer genético)

Mejoras:
- Soporte robusto para custom_track_data (surfaces escaladas/convertidas)
- Detección de checkpoints y finish line (cruces de segmentos + dirección)
- Detección de zonas (speed/slow) con surfaces opcionales (fallback por color en track)
- Colisión robusta (centro + esquinas rotadas)
- Observación y step() compatibles con trainer
"""

import pygame
import math
import numpy as np
from car import Car

class Environment:
    def __init__(self, width=1920, height=1080, custom_track_data=None):
        pygame.init()
        self.width = int(width)
        self.height = int(height)

        # Datos provistos por el editor (pueden ser Surfaces o None)
        self.custom_track_data = custom_track_data

        # Inicializar capas (se harán convert/scale si vienen desde editor)
        self.track = None
        self.finish_line = None
        self.checkpoints = []
        self.speed_zones = None
        self.slow_zones = None
        self.required_laps = 1

        if custom_track_data:
            # Se espera que custom_track_data contenga Surfaces (track_layer, speed_zones, slow_zones)
            # o None; además finish_line y checkpoints (tupla/lista).
            # Hacemos defensiva: si la surface no tiene el tamaño correcto, la escalamos.
            try:
                print(f"[Environment] Cargando pista personalizada...")
                print(f"[Environment] Datos recibidos: {list(custom_track_data.keys())}")
                
                t = custom_track_data.get('track_layer')
                if t:
                    print(f"[Environment] ✓ track_layer encontrado: {type(t)}")
                    self.track = self._prepare_surface(t, convert_alpha=False)
                    print(f"[Environment] ✓ track_layer preparado: {self.track.get_size()}")
                else:
                    print(f"[Environment] ✗ track_layer NO encontrado, usando pista por defecto")
                    self.track = self._create_track()

                # IMPORTANTE: Usar coordenadas del JSON, NO las imágenes PNG de líneas
                finish = custom_track_data.get('finish_line')
                if finish:
                    # finish puede venir como (x1,y1,x2,y2,angle) o similar
                    self.finish_line = tuple(finish)
                    print(f"[Environment] ✓ finish_line (coordenadas): {self.finish_line}")
                else:
                    self.finish_line = None
                    print(f"[Environment] ⚠ finish_line no definida")

                cps = custom_track_data.get('checkpoints')
                if cps:
                    # normalizar a lista de tuplas (x1,y1,x2,y2)
                    self.checkpoints = [tuple(cp) for cp in cps]
                    print(f"[Environment] ✓ {len(self.checkpoints)} checkpoints cargados (coordenadas)")
                    for i, cp in enumerate(self.checkpoints):
                        print(f"[Environment]   - Checkpoint {i+1}: ({cp[0]:.1f}, {cp[1]:.1f}) -> ({cp[2]:.1f}, {cp[3]:.1f})")
                else:
                    self.checkpoints = []
                    print(f"[Environment] ⚠ No hay checkpoints definidos")

                sz = custom_track_data.get('speed_zones')
                if sz:
                    self.speed_zones = self._prepare_surface(sz, convert_alpha=True)
                    print(f"[Environment] ✓ speed_zones cargadas")
                else:
                    print(f"[Environment] ⚠ speed_zones no definidas")

                slo = custom_track_data.get('slow_zones')
                if slo:
                    self.slow_zones = self._prepare_surface(slo, convert_alpha=True)
                    print(f"[Environment] ✓ slow_zones cargadas")
                else:
                    print(f"[Environment] ⚠ slow_zones no definidas")

                self.required_laps = int(custom_track_data.get('required_laps', 1))
                print(f"[Environment] ✓ Vueltas requeridas: {self.required_laps}")
                
                # Verificar spawn_point
                spawn = custom_track_data.get('spawn_point')
                if spawn:
                    print(f"[Environment] ✓ Spawn point: ({spawn[0]:.1f}, {spawn[1]:.1f}, {math.degrees(spawn[2]):.1f}°)")
                
                print(f"[Environment] ✓ Pista personalizada cargada exitosamente!")
                
            except Exception as e:
                print(f"[Environment] ✗✗✗ ERROR CRÍTICO al cargar custom_track_data ✗✗✗")
                print(f"[Environment] Error: {e}")
                import traceback
                traceback.print_exc()
                print(f"[Environment] ⚠ FALLBACK: Usando pista procedural por defecto")
                # fallback a pista procedural
                self.track = self._create_track()
                self.finish_line = None
                self.checkpoints = []
                self.speed_zones = None
                self.slow_zones = None
                self.required_laps = 1
        else:
            # Crear pista por defecto
            self.track = self._create_track()
            self.finish_line = None
            self.checkpoints = []
            self.speed_zones = None
            self.slow_zones = None
            self.required_laps = 1

        # Carro asociado al entorno
        self.car = Car(self)

        # Estado del entorno
        self.done = False
        self.action_space = 4

        # Variables para laps / checkpoints (estado global, pero el trainer mantiene por agente)
        self.current_lap = 0
        self.checkpoints_passed = set()
        self.last_finish_cross = None
        self.finish_line_crossed = False

    # -----------------------
    # Helpers para surfaces
    # -----------------------
    def _prepare_surface(self, surf, convert_alpha=False):
        """
        Garantiza que la surface tenga el tamaño (width,height) y el formato adecuado.
        Si surf no es pygame.Surface (por ejemplo, ruta), se intenta cargar con pygame.image.load.
        """
        # Si es string, intentar cargar desde archivo
        try:
            if isinstance(surf, str):
                loaded = pygame.image.load(surf)
            else:
                loaded = surf
        except Exception:
            loaded = surf  # lo dejamos como venga, intentaremos usarlo

        # Algunos objetos podrían ser superficies ya; verificamos
        if isinstance(loaded, pygame.Surface):
            # Escalar si tamaño diferente
            if loaded.get_size() != (self.width, self.height):
                try:
                    loaded = pygame.transform.smoothscale(loaded, (self.width, self.height))
                except Exception:
                    loaded = pygame.transform.scale(loaded, (self.width, self.height))
            # Convertir formato para acelerar get_at
            try:
                if convert_alpha:
                    loaded = loaded.convert_alpha()
                else:
                    loaded = loaded.convert()
            except Exception:
                pass
            return loaded
        else:
            # No es surface - intentar crear una surface vacía como fallback
            s = pygame.Surface((self.width, self.height))
            s.fill((0, 0, 0))
            return s

    # -----------------------
    # Pista por defecto (procedural)
    # -----------------------
    def _create_track(self):
        track = pygame.Surface((self.width, self.height))
        track = track.convert()
        track.fill((0, 0, 0))

        margin_x = int(self.width * 0.1)
        margin_y = int(self.height * 0.1)

        exterior_rect = pygame.Rect(margin_x, margin_y, int(self.width * 0.8), int(self.height * 0.7))
        pygame.draw.ellipse(track, (255, 255, 255), exterior_rect)

        inner_margin_x = int(self.width * 0.2)
        inner_margin_y = int(self.height * 0.25)
        interior_rect = pygame.Rect(inner_margin_x, inner_margin_y, int(self.width * 0.6), int(self.height * 0.5))
        pygame.draw.ellipse(track, (0, 0, 0), interior_rect)

        return track

    # -----------------------
    # Gym-like API
    # -----------------------
    def reset(self):
        """
        Resetea el carro y estados relacionados.
        Devuelve observación: sensores normalizados (array  of floats).
        """
        self.car.reset()
        self.done = False

        # reset local
        self.current_lap = 0
        self.checkpoints_passed = set()
        self.last_finish_cross = None
        self.finish_line_crossed = False

        obs = self.car.get_sensor_distances(self.track)
        observation = np.array(obs, dtype=np.float32) / self.car.max_sensor_distance
        return observation

    def step(self, action):
        """
        Ejecuta acción, actualiza auto y retorna (obs, reward, done, info).
        action: int 0..3 o tupla/lista (steering, throttle)
        """
        prev_pos = (self.car.x, self.car.y)

        # Mapear acciones discretas a (steer, throttle)
        if isinstance(action, int):
            if not (0 <= action < 4):
                raise ValueError(f"Acción inválida: {action}")
            actions_map = [
                (0, 0),   # 0: nada
                (0, 1),   # 1: acel
                (-1, 1),  # 2: izquierda + acel
                (1, 1)    # 3: derecha + acel
            ]
            action = actions_map[action]
        elif isinstance(action, (list, tuple)):
            if len(action) != 2:
                raise ValueError("Acción debe tener 2 elementos (steer, throttle)")
        else:
            raise TypeError("Acción debe ser int o tupla/lista")

        # Aplicar acción y actualizar vehicle
        self.car.apply_action(action)
        self.car.update()

        curr_pos = (self.car.x, self.car.y)

        obs = self.car.get_sensor_distances(self.track)
        observation = np.array(obs, dtype=np.float32) / self.car.max_sensor_distance

        collision = self._check_collision()

        lap_completed = False
        checkpoint_passed = False

        if collision:
            reward = -100.0
            self.done = True
        else:
            # Recompensa base por velocidad (normalizada)
            # Si car.max_speed == 0 (defecto raro), protegemos
            try:
                base_speed_reward = (self.car.speed / self.car.max_speed) if self.car.max_speed != 0 else 0.0
            except Exception:
                base_speed_reward = 0.0
            reward = base_speed_reward
            reward += 0.5  # bono de supervivencia por paso

            # Checkpoints
            if self.checkpoints:
                for i, cp in enumerate(self.checkpoints):
                    if i not in self.checkpoints_passed:
                        if self._check_checkpoint_crossing(i, prev_pos, curr_pos):
                            self.checkpoints_passed.add(i)
                            checkpoint_passed = True
                            reward += 10.0
                            # Debug
                            # print(f"[Env] Checkpoint {i} pasado por auto en {curr_pos}")

            # Finish line
            if self.finish_line:
                if self._check_finish_line_crossing(prev_pos, curr_pos):
                    # Solo si pasó todos los checkpoints
                    if len(self.checkpoints_passed) == len(self.checkpoints):
                        self.current_lap += 1
                        lap_completed = True
                        reward += 50.0
                        # reset checkpoints
                        self.checkpoints_passed = set()
                        # debug
                        # print(f"[Env] Vuelta completada: {self.current_lap}/{self.required_laps}")

                        if self.current_lap >= self.required_laps:
                            reward += 100.0
                            self.done = True
                    else:
                        # cruzó meta sin checkpoints -> pequeño bonus
                        reward += 5.0

            # Penalización por proximidad a bordes (sensores)
            try:
                min_sensor_dist = min(obs)
            except Exception:
                min_sensor_dist = 0.0
            if min_sensor_dist < self.car.max_sensor_distance * 0.2:
                reward -= 0.5
            elif min_sensor_dist > self.car.max_sensor_distance * 0.5:
                reward += 0.2

            # Aplicar efecto de zona (si existe)
            zone_effect = self._get_zone_effect(curr_pos)
            # Evitamos modificar de forma acumulativa la velocidad de manera incontrolada:
            # aplicamos multiplicador temporal sobre speed para calcular reward/efecto, y lo dejamos
            # en un rango sensato.
            try:
                if zone_effect != 1.0:
                    # limitar efecto de velocidad para que no exceda max_speed
                    new_speed = self.car.speed * zone_effect
                    self.car.speed = max(0.0, min(new_speed, getattr(self.car, 'max_speed', new_speed)))
            except Exception:
                pass

        info = {
            'position': (self.car.x, self.car.y),
            'angle': getattr(self.car, 'angle', 0.0),
            'speed': getattr(self.car, 'speed', 0.0),
            'min_sensor': min(obs) if not collision else 0.0,
            'current_lap': self.current_lap,
            'checkpoints_passed': len(self.checkpoints_passed),
            'lap_completed': lap_completed,
            'checkpoint_passed': checkpoint_passed
        }

        return observation, reward, self.done, info

    # -----------------------
    # Colisión robusta
    # -----------------------
    def _check_collision(self):
        """
        Chequea colisión leyendo colores de píxeles.
        Usa color[:3] para ser robusto frente a surfaces con alfa.
        Revisa centro + esquinas rotadas para mayor precisión.
        """
        px = int(self.car.x)
        py = int(self.car.y)
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return True

        try:
            color = self.track.get_at((px, py))[:3]
        except Exception:
            # Si falla get_at (por formato), consideramos colisión por seguridad
            return True

        if color == (0, 0, 0):
            return True

        # Revisar esquinas del rectángulo del auto (aprox)
        half_w = getattr(self.car, 'width_car', 10) / 2.0
        half_h = getattr(self.car, 'height_car', 20) / 2.0

        cos_a = math.cos(self.car.angle)
        sin_a = math.sin(self.car.angle)

        corners = [
            (half_w, half_h),
            (-half_w, half_h),
            (half_w, -half_h),
            (-half_w, -half_h)
        ]

        for dx, dy in corners:
            rx = self.car.x + dx * cos_a - dy * sin_a
            ry = self.car.y + dx * sin_a + dy * cos_a
            ix = int(rx)
            iy = int(ry)
            if ix < 0 or ix >= self.width or iy < 0 or iy >= self.height:
                return True
            try:
                c = self.track.get_at((ix, iy))[:3]
            except Exception:
                return True
            if c == (0, 0, 0):
                return True

        return False

    # -----------------------
    # Line crossing utilities
    # -----------------------
    def _check_line_crossing(self, line, prev_pos, curr_pos):
        """
        Detección de intersección entre segmento prev_pos-curr_pos y línea x1y1-x2y2.
        """
        if not line:
            return False

        x1, y1, x2, y2 = line[:4]
        A = (x1, y1)
        B = (x2, y2)
        C = (prev_pos[0], prev_pos[1])
        D = (curr_pos[0], curr_pos[1])

        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])

        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

    def _check_finish_line_crossing(self, prev_pos, curr_pos):
        """
        Cruce de línea de meta con verificación de dirección.
        Retorna True solo si cruza en la dirección correcta (producto cruz positivo).
        """
        if not self.finish_line:
            return False

        if self._check_line_crossing(self.finish_line, prev_pos, curr_pos):
            x1, y1, x2, y2, *rest = self.finish_line
            # vector línea
            line_dx = x2 - x1
            line_dy = y2 - y1
            # vector movimiento
            move_dx = curr_pos[0] - prev_pos[0]
            move_dy = curr_pos[1] - prev_pos[1]
            cross = line_dx * move_dy - line_dy * move_dx
            return cross > 0
        return False

    def _check_checkpoint_crossing(self, checkpoint_idx, prev_pos, curr_pos):
        if checkpoint_idx < 0 or checkpoint_idx >= len(self.checkpoints):
            return False
        cp = self.checkpoints[checkpoint_idx]
        return self._check_line_crossing(cp, prev_pos, curr_pos)

    # -----------------------
    # Zonas por color
    # -----------------------
    def _get_zone_effect(self, pos):
        """
        Retorna multiplicador de velocidad según zona:
         - speed zone (verde): 1.5
         - slow zone (amarillo): 0.7
         - fallback: 1.0
        Prioriza surfaces separadas (speed_zones/slow_zones), si no existen usa color en track.
        """
        px = int(max(0, min(self.width - 1, pos[0])))
        py = int(max(0, min(self.height - 1, pos[1])))

        # speed_zones surface
        if self.speed_zones:
            try:
                c = self.speed_zones.get_at((px, py))[:3]
                if c == (0, 255, 0):
                    return 1.5
            except Exception:
                pass

        if self.slow_zones:
            try:
                c = self.slow_zones.get_at((px, py))[:3]
                if c == (255, 255, 0):
                    return 0.7
            except Exception:
                pass

        # Fallback a color en track (combinada)
        try:
            c = self.track.get_at((px, py))[:3]
            if c == (0, 255, 0):
                return 1.5
            if c == (255, 255, 0):
                return 0.7
        except Exception:
            pass

        return 1.0

    # -----------------------
    # Render
    # -----------------------
    def render(self, screen):
        """
        Renderiza la pista, capas especiales, checkpoints, finish line y el auto.
        """
        if self.track:
            try:
                screen.blit(self.track, (0, 0))
            except Exception:
                # si falla blit, rellenamos fondo de forma segura
                screen.fill((0, 0, 0))
        else:
            screen.fill((0, 0, 0))

        # capas especiales
        if self.speed_zones:
            try:
                screen.blit(self.speed_zones, (0, 0))
            except Exception:
                pass
        if self.slow_zones:
            try:
                screen.blit(self.slow_zones, (0, 0))
            except Exception:
                pass

        # Draw checkpoints (si existen)
        if self.checkpoints:
            for i, cp in enumerate(self.checkpoints):
                color = (0, 255, 0) if i in self.checkpoints_passed else (0, 100, 255)
                pygame.draw.line(screen, color, (cp[0], cp[1]), (cp[2], cp[3]), 8)

        # Draw finish line
        if self.finish_line:
            try:
                fx1, fy1, fx2, fy2 = self.finish_line[:4]
                pygame.draw.line(screen, (255, 0, 0), (fx1, fy1), (fx2, fy2), 10)
            except Exception:
                pass

        # Draw car + sensors using Car class methods
        try:
            self.car.draw(screen)
            self.car.draw_sensors(screen)
        except Exception:
            pass

        # Mostrar info de vueltas
        if self.finish_line:
            try:
                font = pygame.font.Font(None, 28)
                lap_text = font.render(f"Vuelta: {self.current_lap}/{self.required_laps}", True, (255, 255, 255))
                pygame.draw.rect(screen, (0, 0, 0), (10, 50, 250, 35))
                screen.blit(lap_text, (15, 55))
            except Exception:
                pass
