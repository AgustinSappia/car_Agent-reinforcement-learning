"""
train_genetic.py - Versión mejorada
Integración con Track Editor:
- Detección de línea de meta y conteo de vueltas
- Detección de checkpoints (cruce de segmentos)
- Detección de zonas rápidas/lentas por color
- Recompensas actualizadas
- Manejo por agente de progreso (laps, checkpoints pasados, prev_pos)
"""

import torch
import numpy as np
import random
from collections import deque
import pygame
import os
import csv
from copy import deepcopy
from model import LinearQNet
from environment import Environment

# Colores para el menú
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)

# Colores para los agentes
AGENT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


class GeneticAgent:
    """Agente individual con su propia red neuronal"""
    def __init__(self, state_size, action_size, agent_id, color, learning_rate=0.001):
        self.agent_id = agent_id
        self.color = color
        self.state_size = state_size
        self.action_size = action_size

        # Red neuronal
        self.brain = LinearQNet(state_size, output_size=action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain.to(self.device)

        # Métricas
        self.fitness = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.is_alive = True

        # Posición y estado del auto (solo para render)
        self.x = 0
        self.y = 0
        self.angle = 0
        self.speed = 0

    def act(self, state, epsilon=0.0):
        """Selecciona acción (con exploración opcional)"""
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.brain(state_tensor)
        return np.argmax(q_values.cpu().data.numpy()).item()

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        """Muta los pesos de la red neuronal"""
        with torch.no_grad():
            for param in self.brain.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)

    def clone(self):
        """Crea una copia del agente"""
        new_agent = GeneticAgent(self.state_size, self.action_size, self.agent_id, self.color)
        new_agent.brain.load_state_dict(deepcopy(self.brain.state_dict()))
        return new_agent

    def save(self, filename):
        """Guarda el cerebro del agente"""
        torch.save(self.brain.state_dict(), filename)

    def load(self, filename):
        """Carga el cerebro del agente"""
        if os.path.exists(filename):
            self.brain.load_state_dict(torch.load(filename, map_location=self.device))
            return True
        return False


class ControlMenu:
    """Menú interactivo para controlar parámetros del entrenamiento"""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # Parámetros ajustables
        self.params = {
            'num_agents': {'value': 5, 'min': 1, 'max': 100, 'step': 1, 'label': 'Agentes'},
            'max_reward': {'value': 1000, 'min': 100, 'max': 500000, 'step': 200, 'label': 'Max Reward'},
            'epsilon': {'value': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'label': 'Epsilon'},
            'speed_mult': {'value': 1.0, 'min': 0.5, 'max': 5.0, 'step': 0.5, 'label': 'Velocidad'},
        }

        self.selected_param = None
        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 24)

    def draw(self, screen, generation, best_fitness):
        """Dibuja el menú en la pantalla"""
        # Fondo del menú
        menu_surface = pygame.Surface((self.width, self.height))
        menu_surface.set_alpha(220)
        menu_surface.fill((40, 40, 40))
        screen.blit(menu_surface, (self.x, self.y))

        # Borde
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height), 2)

        # Título
        title = self.title_font.render("CONTROL PANEL", True, YELLOW)
        screen.blit(title, (self.x + 10, self.y + 10))

        # Información de generación
        gen_text = self.font.render(f"Generación: {generation}", True, WHITE)
        screen.blit(gen_text, (self.x + 10, self.y + 40))

        fitness_text = self.font.render(f"Mejor Fitness: {best_fitness:.1f}", True, GREEN)
        screen.blit(fitness_text, (self.x + 10, self.y + 60))

        # Parámetros ajustables
        y_offset = 90
        for key, param in self.params.items():
            # Etiqueta
            label = self.font.render(f"{param['label']}:", True, WHITE)
            screen.blit(label, (self.x + 10, self.y + y_offset))

            # Valor actual
            value_text = f"{param['value']:.2f}" if isinstance(param['value'], float) else str(param['value'])
            value = self.font.render(value_text, True, YELLOW)
            screen.blit(value, (self.x + 120, self.y + y_offset))

            # Botones - y +
            minus_rect = pygame.Rect(self.x + 180, self.y + y_offset - 2, 25, 20)
            plus_rect = pygame.Rect(self.x + 210, self.y + y_offset - 2, 25, 20)

            pygame.draw.rect(screen, RED, minus_rect)
            pygame.draw.rect(screen, GREEN, plus_rect)
            pygame.draw.rect(screen, WHITE, minus_rect, 1)
            pygame.draw.rect(screen, WHITE, plus_rect, 1)

            minus_text = self.font.render("-", True, WHITE)
            plus_text = self.font.render("+", True, WHITE)
            screen.blit(minus_text, (minus_rect.x + 8, minus_rect.y + 2))
            screen.blit(plus_text, (plus_rect.x + 7, plus_rect.y + 2))

            # Guardar rects para detección de clicks
            param['minus_rect'] = minus_rect
            param['plus_rect'] = plus_rect

            y_offset += 30

        # Instrucciones
        inst_y = self.y + self.height - 60
        inst1 = self.font.render("Click +/- para ajustar", True, LIGHT_GRAY)
        inst2 = self.font.render("ESC: Salir | SPACE: Pausa", True, LIGHT_GRAY)
        screen.blit(inst1, (self.x + 10, inst_y))
        screen.blit(inst2, (self.x + 10, inst_y + 20))

    def handle_click(self, pos):
        """Maneja clicks en el menú"""
        for key, param in self.params.items():
            if 'minus_rect' in param and param['minus_rect'].collidepoint(pos):
                step = param['step']
                if key == 'epsilon':
                    step = 0.01 if param['value'] < 0.1 else 0.1

                new_value = param['value'] - step
                param['value'] = max(param['min'], new_value)
                return key, param['value']

            if 'plus_rect' in param and param['plus_rect'].collidepoint(pos):
                step = param['step']
                if key == 'epsilon':
                    step = 0.01 if param['value'] < 0.1 else 0.1

                new_value = param['value'] + step
                param['value'] = min(param['max'], new_value)
                return key, param['value']

        return None, None


class GeneticTrainer:
    """Sistema de entrenamiento genético/evolutivo"""
    def __init__(self, width=1920, height=1080, custom_track_data=None):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Genetic Training - Self Driving Car AI")
        self.clock = pygame.time.Clock()

        # Soporte para pista personalizada (datos provistos por el editor)
        self.custom_track_data = custom_track_data

        # Entorno compartido (Environment debe aceptar custom_track_data)
        self.env = Environment(width=width, height=height, custom_track_data=custom_track_data)

        # Configuración
        self.state_size = 7
        self.action_size = 4
        self.generation = 1
        self.best_fitness_ever = 0.0
        self.best_agent_ever = None

        # Menú de control
        menu_width = 250
        menu_height = 300
        self.menu = ControlMenu(width - menu_width - 20, 20, menu_width, menu_height)

        # Agentes
        self.agents = []
        self.create_agents(self.menu.params['num_agents']['value'])

        # Mapping de acciones
        self.action_map = [
            [1, 0],  # 0: nada
            [1, 1],  # 1: acelerar
            [0, 1],  # 2: izquierda + acel
            [2, 1]   # 3: derecha + acel
        ]

        # Estados de los agentes
        self.agent_states = {}

        # Control
        self.paused = False
        self.running = True

        # Log
        self.log_file = 'genetic_training_log.csv'
        self.init_log()

    def init_log(self):
        """Inicializa el archivo de log"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Best_Fitness', 'Avg_Fitness', 'Num_Agents', 'Epsilon'])

    def create_agents(self, num_agents):
        """Crea la población inicial de agentes"""
        self.agents = []
        for i in range(num_agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            agent = GeneticAgent(self.state_size, self.action_size, i, color)
            self.agents.append(agent)

        # Intentar cargar el mejor agente
        if os.path.exists('best_genetic_agent.pth'):
            self.agents[0].load('best_genetic_agent.pth')
            print("✓ Mejor agente cargado")

    def reset_generation(self):
        """Reinicia todos los agentes para una nueva generación"""
        for i, agent in enumerate(self.agents):
            agent.fitness = 0.0
            agent.total_reward = 0.0
            agent.steps = 0
            agent.is_alive = True

            # Crear estado inicial para cada agente con info de progreso
            car = self.create_car_for_agent()
            spawn_point = (car.x, car.y)
            self.agent_states[i] = {
                'car': car,
                'done': False,
                'prev_pos': (car.x, car.y) if hasattr(car, 'x') else (0, 0),
                'spawn_point': spawn_point,
                'max_distance': 0.0,  # Máxima distancia alcanzada desde spawn
                'last_progress_step': 0,  # Último step donde hubo progreso
                'last_position': (car.x, car.y),  # Para detectar si está atascado
                'stuck_counter': 0,  # Contador de steps sin movimiento significativo
                'laps': 0,
                'passed_checkpoints': [False] * (len(self.env.checkpoints) if hasattr(self.env, 'checkpoints') else 0),
                'last_finish_cross_step': -100,  # Para evitar spam de cruces
            }

    def create_car_for_agent(self):
        """Crea un auto independiente para un agente"""
        from car import Car
        car = Car(self.env)
        car.reset()
        # Dar velocidad inicial moderada para moverse
        try:
            car.speed = 3.0
        except Exception:
            pass

        # Si existe spawn point en custom_track_data, colocar el auto ahí
        if self.custom_track_data and 'spawn_point' in self.custom_track_data and self.custom_track_data['spawn_point']:
            sp = self.custom_track_data['spawn_point']
            try:
                # spawn_point: (x, y, angle)
                car.x = sp[0]
                car.y = sp[1]
                car.angle = sp[2]
            except Exception:
                pass

        return car

    # -----------------------
    # UTIL: Intersección de segmentos (para checkpoints / finish line)
    # -----------------------
    def _orient(self, a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _on_segment(self, a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    def segment_intersect(self, p1, p2, q1, q2):
        """Devuelve True si segmentos p1-p2 y q1-q2 se intersectan"""
        o1 = self._orient(p1, p2, q1)
        o2 = self._orient(p1, p2, q2)
        o3 = self._orient(q1, q2, p1)
        o4 = self._orient(q1, q2, p2)

        if o1 == 0 and self._on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self._on_segment(p1, p2, q2):
            return True
        if o3 == 0 and self._on_segment(q1, q2, p1):
            return True
        if o4 == 0 and self._on_segment(q1, q2, p2):
            return True

        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    # -----------------------
    # PROGRESS & ZONES CHECK
    # -----------------------
    def _cross_product_2d(self, v1, v2):
        """Producto cruz 2D para determinar dirección"""
        return v1[0] * v2[1] - v1[1] * v2[0]
    
    def check_progress_and_zones(self, car, state, agent_idx):
        """
        Revisa:
        - Cruce de checkpoints (y marcarlos para esta vuelta)
        - Cruce de línea de meta CON DIRECCIÓN (y contar vuelta si pasaron todos los checkpoints)
        - Detección de zona rápida / zona lenta por color
        - Progreso real (distancia desde spawn)
        Devuelve (reward_bonus, done_flag, info_dict)
        """
        reward = 0.0
        done = False
        info = {}

        st = self.agent_states[agent_idx]
        prev_pos = st.get('prev_pos', (car.x, car.y))
        cur_pos = (car.x, car.y)
        spawn_point = st.get('spawn_point', (0, 0))

        # 1) Recompensa por PROGRESO REAL (alejarse del spawn)
        current_distance = np.sqrt((cur_pos[0] - spawn_point[0])**2 + (cur_pos[1] - spawn_point[1])**2)
        if current_distance > st['max_distance']:
            # Nuevo territorio explorado
            progress_reward = (current_distance - st['max_distance']) * 0.1
            reward += progress_reward
            st['max_distance'] = current_distance
            st['last_progress_step'] = self.agents[agent_idx].steps
            info['progress'] = True

        # 2) Penalización por quedarse quieto (sin progreso)
        steps_since_progress = self.agents[agent_idx].steps - st['last_progress_step']
        if steps_since_progress > 100:  # 100 steps sin progreso
            reward -= 1.0
            info['stagnant'] = True
        
        # 2b) Detectar si está atascado (girando en el mismo lugar)
        last_pos = st.get('last_position', cur_pos)
        distance_moved = np.sqrt((cur_pos[0] - last_pos[0])**2 + (cur_pos[1] - last_pos[1])**2)
        
        if distance_moved < 5.0:  # Se movió menos de 5 píxeles
            st['stuck_counter'] += 1
            if st['stuck_counter'] > 50:  # 50 steps sin moverse significativamente
                reward -= 2.0  # Penalización fuerte
                info['stuck'] = True
                # Si lleva mucho tiempo atascado, marcarlo como muerto
                if st['stuck_counter'] > 150:  # 150 steps atascado = muerte
                    done = True
                    reward -= 100.0
                    info['timeout_stuck'] = True
        else:
            st['stuck_counter'] = 0  # Reset si se movió
        
        st['last_position'] = cur_pos

        # 3) Checkpoints: si existen en el env
        checkpoints = getattr(self.env, 'checkpoints', None)
        if checkpoints:
            for cp_idx, cp in enumerate(checkpoints):
                q1 = (cp[0], cp[1])
                q2 = (cp[2], cp[3])
                if not st['passed_checkpoints'][cp_idx]:
                    # si el movimiento del auto cruza el segmento del checkpoint
                    if self.segment_intersect(prev_pos, cur_pos, q1, q2):
                        st['passed_checkpoints'][cp_idx] = True
                        reward += 100.0  # GRAN bono por pasar checkpoint
                        st['last_progress_step'] = self.agents[agent_idx].steps
                        info.setdefault('checkpoints', []).append(cp_idx)

        # 4) Finish line CON DETECCIÓN DE DIRECCIÓN: si existe
        finish = getattr(self.env, 'finish_line', None)
        required_laps = getattr(self.env, 'required_laps', None) or (self.custom_track_data.get('required_laps') if self.custom_track_data else 0)
        if finish:
            f1 = (finish[0], finish[1])
            f2 = (finish[2], finish[3])
            
            # Evitar spam de cruces (cooldown de 50 steps)
            current_step = self.agents[agent_idx].steps
            if self.segment_intersect(prev_pos, cur_pos, f1, f2) and (current_step - st['last_finish_cross_step']) > 50:
                st['last_finish_cross_step'] = current_step
                
                # Calcular dirección del cruce
                # Vector de la línea de meta
                finish_vec = (f2[0] - f1[0], f2[1] - f1[1])
                # Vector de movimiento del auto
                move_vec = (cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1])
                # Producto cruz para determinar lado
                cross = self._cross_product_2d(finish_vec, move_vec)
                
                # Si hay ángulo en metadata, usarlo para determinar dirección correcta
                correct_direction = True
                if len(finish) > 4:
                    # Asumimos que cruzar en sentido positivo (cross > 0) es correcto
                    correct_direction = cross > 0
                
                if correct_direction:
                    # Verificar si se completaron todos los checkpoints (si hay)
                    all_cp_passed = True
                    if checkpoints:
                        all_cp_passed = all(st['passed_checkpoints'])
                    
                    if all_cp_passed:
                        st['laps'] += 1
                        reward += 200.0  # GRAN bono por cruzar meta correctamente
                        # resetear checkpoints para la siguiente vuelta
                        st['passed_checkpoints'] = [False] * len(st['passed_checkpoints'])
                        st['last_progress_step'] = current_step
                        info['lap'] = st['laps']
                        # Si alcanza laps requeridas -> done
                        if required_laps and st['laps'] >= required_laps:
                            done = True
                            reward += 1000.0  # ENORME recompensa por completar
                            info['completed'] = True
                    else:
                        # Cruzó sin checkpoints
                        reward -= 50.0  # Penalización por hacer trampa
                        info['lap_invalid'] = True
                else:
                    # Cruzó en dirección INCORRECTA
                    reward -= 100.0  # GRAN penalización
                    info['wrong_direction'] = True

        # 5) Zonas por color: SOLO pequeño bonus/malus
        px = int(max(0, min(self.env.track.get_width()-1, int(car.x))))
        py = int(max(0, min(self.env.track.get_height()-1, int(car.y))))

        speed_zones = getattr(self.env, 'speed_zones', None)
        slow_zones = getattr(self.env, 'slow_zones', None)
        try:
            if speed_zones and speed_zones.get_at((px, py))[:3] == (0, 255, 0):
                reward += 0.5  # Pequeño bonus
                info['zone'] = 'speed'
            elif slow_zones and slow_zones.get_at((px, py))[:3] == (255, 255, 0):
                reward -= 0.3  # Pequeña penalización
                info['zone'] = 'slow'
            else:
                # fallback a color directo en track combinado
                color_at = self.env.track.get_at((px, py))[:3]
                if color_at == (0, 255, 0):
                    reward += 0.5
                    info['zone'] = 'speed'
                elif color_at == (255, 255, 0):
                    reward -= 0.3
                    info['zone'] = 'slow'
        except Exception:
            pass

        # Actualizar prev_pos
        st['prev_pos'] = cur_pos

        return reward, done, info

    # -----------------------
    # COLISIÓN / FUERA DE PISTA
    # -----------------------
    def check_collision(self, car):
        """Verifica colisión del auto.
           Considera off-track si el pixel es negro (0,0,0).
           NOTA: algunos mapas pueden tener alfa; por eso usamos [:3].
        """
        px, py = int(car.x), int(car.y)
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return True

        try:
            color = self.env.track.get_at((px, py))[:3]
        except Exception:
            # si falla lectura (ej: surfaces distintas), marcar como colisión por seguridad
            return True

        # Si la capa finish/checkpoint/zones están pintadas en colores distintos, no son colisión.
        if color == BLACK:
            return True

        # En cualquier otro color (blanco, verde, amarillo, rojo, azul) no es colisión.
        return False

    # -----------------------
    # RENDER & LOOP PRINCIPAL
    # -----------------------
    def render(self):
        """Renderiza el entorno y los agentes"""
        # Dibujar pista base
        self.screen.blit(self.env.track, (0, 0))

        # Dibujar capas especiales (zonas de velocidad) con blending especial
        # Solo dibujamos los píxeles de color, no el fondo negro
        if self.env.speed_zones:
            try:
                # Crear una copia temporal para aplicar blending
                temp_surface = self.env.speed_zones.copy()
                temp_surface.set_colorkey((0, 0, 0))  # Hacer transparente el negro
                self.screen.blit(temp_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
            except Exception as e:
                pass
        
        if self.env.slow_zones:
            try:
                # Crear una copia temporal para aplicar blending
                temp_surface = self.env.slow_zones.copy()
                temp_surface.set_colorkey((0, 0, 0))  # Hacer transparente el negro
                self.screen.blit(temp_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
            except Exception as e:
                pass

        # Dibujar checkpoints (si existen)
        if self.env.checkpoints:
            for i, cp in enumerate(self.env.checkpoints):
                # Color verde si ya fue pasado, azul si no
                # Nota: esto es global, no por agente individual
                color = (0, 100, 255)  # Azul por defecto
                try:
                    pygame.draw.line(self.screen, color, (cp[0], cp[1]), (cp[2], cp[3]), 8)
                except Exception:
                    pass

        # Dibujar línea de meta (finish line)
        if self.env.finish_line:
            try:
                fx1, fy1, fx2, fy2 = self.env.finish_line[:4]
                pygame.draw.line(self.screen, RED, (fx1, fy1), (fx2, fy2), 10)
            except Exception:
                pass

        # Dibujar agentes
        for agent in self.agents:
            if agent.is_alive:
                # Dibujar auto (superficie simple)
                car_surf = pygame.Surface((20, 40))
                car_surf.fill(agent.color)
                rotated_surf = pygame.transform.rotate(car_surf, -np.degrees(agent.angle))
                rotated_rect = rotated_surf.get_rect(center=(int(agent.x), int(agent.y)))
                self.screen.blit(rotated_surf, rotated_rect.topleft)

                # Etiqueta con fitness
                font = pygame.font.Font(None, 16)
                label = font.render(f"#{agent.agent_id}: {agent.fitness:.0f}", True, WHITE)
                self.screen.blit(label, (int(agent.x) - 20, int(agent.y) - 30))

        # Mostrar info de vueltas si hay finish line
        if self.env.finish_line:
            try:
                font = pygame.font.Font(None, 28)
                # Mostrar vueltas del primer agente vivo como referencia
                laps_text = "Vueltas: 0/0"
                for i, agent in enumerate(self.agents):
                    if agent.is_alive and i in self.agent_states:
                        st = self.agent_states[i]
                        laps_text = f"Vueltas: {st['laps']}/{self.env.required_laps}"
                        break
                lap_surface = font.render(laps_text, True, WHITE)
                pygame.draw.rect(self.screen, BLACK, (10, 50, 250, 35))
                self.screen.blit(lap_surface, (15, 55))
            except Exception:
                pass

        # Dibujar menú
        best_fitness = max([a.fitness for a in self.agents]) if self.agents else 0
        self.menu.draw(self.screen, self.generation, best_fitness)

        # Indicador de pausa
        if self.paused:
            font = pygame.font.Font(None, 72)
            pause_text = font.render("PAUSA", True, YELLOW)
            text_rect = pause_text.get_rect(center=(self.width // 2, self.height // 2))
            pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 20))
            self.screen.blit(pause_text, text_rect)

        pygame.display.flip()

    def select_best_and_evolve(self):
        """Selecciona los mejores agentes y crea nueva generación"""
        # Ordenar por fitness
        self.agents.sort(key=lambda a: a.fitness, reverse=True)

        best_fitness = self.agents[0].fitness
        avg_fitness = np.mean([a.fitness for a in self.agents])

        print(f"\n{'='*60}")
        print(f"GENERACIÓN {self.generation} COMPLETADA")
        print(f"{'='*60}")
        print(f"Mejor Fitness: {best_fitness:.2f}")
        print(f"Fitness Promedio: {avg_fitness:.2f}")
        print(f"{'='*60}\n")

        # Guardar mejor agente de todos los tiempos
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_agent_ever = self.agents[0].clone()
            self.agents[0].save('best_genetic_agent.pth')
            print(f"✓ Nuevo mejor agente guardado! Fitness: {best_fitness:.2f}\n")

        # Log
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.generation,
                best_fitness,
                avg_fitness,
                len(self.agents),
                self.menu.params['epsilon']['value']
            ])

        # Crear nueva generación
        num_agents = self.menu.params['num_agents']['value']
        new_agents = []

        # Más elitismo
        num_elite = max(2, num_agents // 3)  # 33% elite

        # Mantener los mejores (elitismo)
        for i in range(num_elite):
            if i < len(self.agents):
                elite = self.agents[i].clone()
                elite.agent_id = i
                elite.color = AGENT_COLORS[i % len(AGENT_COLORS)]
                new_agents.append(elite)

        # Mutación adaptativa
        mutation_strength = 0.5 if best_fitness < -80 else 0.3
        mutation_rate = 0.3 if best_fitness < -80 else 0.2

        while len(new_agents) < num_agents:
            parent_idx = random.randint(0, min(num_elite - 1, len(self.agents) - 1))
            child = self.agents[parent_idx].clone()

            child.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            child.agent_id = len(new_agents)
            child.color = AGENT_COLORS[len(new_agents) % len(AGENT_COLORS)]
            new_agents.append(child)

        self.agents = new_agents
        self.generation += 1

    def run(self):
        """Loop principal de entrenamiento"""
        self.reset_generation()

        print("\n" + "="*60)
        print("ENTRENAMIENTO GENÉTICO INICIADO")
        print("="*60)
        print(f"Agentes por generación: {len(self.agents)}")
        print(f"Resolución: {self.width}x{self.height}")
        print("="*60 + "\n")

        while self.running:
            # Procesar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Click izquierdo
                        param_key, new_value = self.menu.handle_click(event.pos)
                        if param_key == 'num_agents' and new_value != len(self.agents):
                            # Reiniciar con nuevo número de agentes
                            self.create_agents(int(new_value))
                            self.reset_generation()

            if not self.paused:
                # Actualizar agentes
                speed_mult = self.menu.params['speed_mult']['value']
                epsilon = self.menu.params['epsilon']['value']

                for _ in range(int(speed_mult)):
                    all_done = True

                    for i, agent in enumerate(self.agents):
                        if not agent.is_alive:
                            continue

                        all_done = False
                        state_data = self.agent_states[i]
                        car = state_data['car']

                        # Obtener observación
                        sensor_distances = car.get_sensor_distances(self.env.track)
                        obs = np.array(sensor_distances, dtype=np.float32) / car.max_sensor_distance
                        speed_norm = car.speed / car.max_speed
                        angle_norm = (car.angle + np.pi) / (2 * np.pi)
                        state = np.array(list(obs) + [speed_norm, angle_norm], dtype=np.float32)

                        # Seleccionar acción
                        action_idx = agent.act(state, epsilon)
                        action = self.action_map[action_idx]

                        # Aplicar acción y actualizar física
                        car.apply_action(action)
                        car.update()

                        # Verificar colisión
                        collision = self.check_collision(car)

                        # Recompensa base
                        if collision:
                            reward = -200.0  # Mayor penalización por colisión
                            agent.is_alive = False
                        else:
                            # NO dar recompensa por solo moverse
                            reward = 0.0
                            
                            # Pequeña recompensa por velocidad (incentiva moverse)
                            reward += (car.speed / car.max_speed) * 0.1

                            # Proximidad a bordes (con sensores) - penalización más fuerte
                            min_sensor = min(sensor_distances)
                            if min_sensor < car.max_sensor_distance * 0.15:
                                reward -= 1.0  # Penalización fuerte por estar muy cerca
                            elif min_sensor < car.max_sensor_distance * 0.3:
                                reward -= 0.3

                            # Check progress (checkpoints, finish line, zones, distancia)
                            prog_reward, done_flag, info = self.check_progress_and_zones(car, state, i)
                            reward += prog_reward
                            
                            # Aplicar efecto de zona directamente a la velocidad del carro
                            if 'zone' in info:
                                if info['zone'] == 'speed':
                                    # Zona rápida: aumentar velocidad temporalmente
                                    car.speed = min(car.speed * 1.2, car.max_speed * 1.5)
                                elif info['zone'] == 'slow':
                                    # Zona lenta: reducir velocidad
                                    car.speed = max(car.speed * 0.8, car.max_speed * 0.3)
                            
                            if done_flag:
                                agent.is_alive = False
                                # Ya se dio gran recompensa en check_progress_and_zones

                        agent.total_reward += reward
                        agent.steps += 1
                        # Fitness valuing supervivencia y reward acumulado
                        agent.fitness = agent.total_reward + agent.steps * 0.5

                        # Actualizar posición para render
                        agent.x = car.x
                        agent.y = car.y
                        agent.angle = car.angle
                        agent.speed = car.speed

                        # TIMEOUT GLOBAL: Si un agente lleva demasiado tiempo sin terminar
                        if agent.steps > 2000:  # 2000 steps = timeout
                            agent.is_alive = False
                            agent.total_reward -= 50.0  # Penalización por timeout
                            state_data['done'] = True

                        # Si el agente ya cumplió (is_alive False por done), marcar
                        if not agent.is_alive:
                            state_data['done'] = True

                    # Si todos terminaron, nueva generación
                    if all_done:
                        self.select_best_and_evolve()
                        self.reset_generation()
                        break

            # Renderizar
            self.render()
            self.clock.tick(60)

        pygame.quit()
        print("\n✓ Entrenamiento finalizado")


if __name__ == "__main__":
    # Selector de pistas externo (track_selector + track_loader)
    print("\n" + "="*60)
    print("INICIANDO SELECTOR DE PISTAS")
    print("="*60 + "\n")

    from track_selector import select_track
    from track_loader import load_track_data

    selected_track = select_track()

    custom_track_data = None
    if selected_track:
        print(f"\n{'='*60}")
        print(f"CARGANDO PISTA SELECCIONADA")
        print(f"{'='*60}")
        print(f"Nombre: {selected_track['name']}")
        print(f"Ruta base: {selected_track.get('path', 'N/A')}")
        print(f"{'='*60}\n")
        
        track_data = load_track_data(selected_track['name'])
        
        if track_data:
            # Validar que track_data tenga los campos necesarios
            required_fields = ['track_layer', 'name']
            missing_fields = [field for field in required_fields if field not in track_data]
            
            if missing_fields:
                print(f"\n{'='*60}")
                print(f"✗✗✗ ERROR: DATOS DE PISTA INCOMPLETOS ✗✗✗")
                print(f"{'='*60}")
                print(f"Campos faltantes: {', '.join(missing_fields)}")
                print(f"La pista no se puede cargar correctamente.")
                print(f"Usando pista por defecto...")
                print(f"{'='*60}\n")
                custom_track_data = None
            else:
                custom_track_data = track_data
                print(f"\n{'='*60}")
                print(f"✓✓✓ PISTA CARGADA EXITOSAMENTE ✓✓✓")
                print(f"{'='*60}")
                print(f"Nombre: {selected_track['name']}")
                print(f"Dimensiones: {track_data.get('width', 'N/A')}x{track_data.get('height', 'N/A')}")
                print(f"Vueltas requeridas: {track_data.get('required_laps', 1)}")
                print(f"Checkpoints: {len(track_data.get('checkpoints', []))}")
                print(f"Línea de meta: {'Sí' if track_data.get('finish_line') else 'No'}")
                print(f"Punto de spawn: {'Sí' if track_data.get('spawn_point') else 'No'}")
                print(f"Zonas rápidas: {'Sí' if track_data.get('speed_zones') else 'No'}")
                print(f"Zonas lentas: {'Sí' if track_data.get('slow_zones') else 'No'}")
                print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"✗✗✗ ERROR CARGANDO PISTA ✗✗✗")
            print(f"{'='*60}")
            print(f"No se pudo cargar la pista: {selected_track['name']}")
            print(f"Verifique que los archivos de la pista existan:")
            print(f"  - tracks/{selected_track['name']}.json")
            print(f"  - tracks/{selected_track['name']}_track.png")
            print(f"Usando pista por defecto...")
            print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"NO SE SELECCIONÓ NINGUNA PISTA")
        print(f"{'='*60}")
        print(f"Usando pista procedural por defecto...")
        print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"INICIANDO ENTRENAMIENTO GENÉTICO")
    print(f"{'='*60}")
    if custom_track_data:
        print(f"Modo: Pista personalizada ({custom_track_data.get('name', 'desconocida')})")
    else:
        print(f"Modo: Pista procedural por defecto")
    print(f"{'='*60}\n")

    trainer = GeneticTrainer(width=1920, height=1080, custom_track_data=custom_track_data)
    trainer.run()
