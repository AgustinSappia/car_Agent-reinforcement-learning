"""
Sistema de Entrenamiento Genético/Evolutivo para Self Driving Car
- Múltiples agentes entrenando simultáneamente
- Selección del mejor agente por generación
- Menú interactivo para ajustar parámetros en tiempo real
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
    (255, 0, 0),    # Rojo
    (0, 255, 0),    # Verde
    (0, 0, 255),    # Azul
    (255, 255, 0),  # Amarillo
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Naranja
    (128, 0, 255),  # Púrpura
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
        
        # Posición y estado del auto
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
                # TASK 1: Dynamic step for epsilon - slower when < 0.1, faster when >= 0.1
                step = param['step']
                if key == 'epsilon':
                    step = 0.01 if param['value'] < 0.1 else 0.1
                
                new_value = param['value'] - step
                param['value'] = max(param['min'], new_value)
                return key, param['value']
            
            if 'plus_rect' in param and param['plus_rect'].collidepoint(pos):
                # TASK 1: Dynamic step for epsilon - slower when < 0.1, faster when >= 0.1
                step = param['step']
                if key == 'epsilon':
                    step = 0.01 if param['value'] < 0.1 else 0.1
                
                new_value = param['value'] + step
                param['value'] = min(param['max'], new_value)
                return key, param['value']
        
        return None, None


class GeneticTrainer:
    """Sistema de entrenamiento genético/evolutivo"""
    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Genetic Training - Self Driving Car AI")
        self.clock = pygame.time.Clock()
        
        # Entorno compartido
        self.env = Environment(width=width, height=height)
        
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
            
            # Crear estado inicial para cada agente
            self.agent_states[i] = {
                'car': self.create_car_for_agent(),
                'done': False
            }
    
    def create_car_for_agent(self):
        """Crea un auto independiente para un agente"""
        from car import Car
        car = Car(self.env)
        car.reset()
        # CORREGIDO: Dar velocidad inicial más alta para que se muevan
        car.speed = 3.0  # Velocidad inicial más alta
        return car
    
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
        
        # MEJORADO: Más elitismo y mejor mutación
        num_elite = max(2, num_agents // 3)  # Mantener más elite (33%)
        
        # Mantener los mejores (elitismo)
        for i in range(num_elite):
            if i < len(self.agents):
                elite = self.agents[i].clone()
                elite.agent_id = i
                elite.color = AGENT_COLORS[i % len(AGENT_COLORS)]
                new_agents.append(elite)
        
        # MEJORADO: Crear el resto con mutación más agresiva al inicio
        # Mutación adaptativa: más fuerte si no hay progreso
        mutation_strength = 0.5 if best_fitness < -80 else 0.3
        mutation_rate = 0.3 if best_fitness < -80 else 0.2
        
        while len(new_agents) < num_agents:
            # Seleccionar padre de los mejores
            parent_idx = random.randint(0, min(num_elite - 1, len(self.agents) - 1))
            child = self.agents[parent_idx].clone()
            
            # Mutar con parámetros adaptativos
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
                        
                        # Aplicar acción
                        car.apply_action(action)
                        car.update()
                        
                        # Verificar colisión
                        collision = self.check_collision(car)
                        
                        # MEJORADO: Sistema de recompensas que fomenta exploración
                        if collision:
                            # Penalización menos severa para fomentar intentos
                            reward = -50.0
                            agent.is_alive = False
                        else:
                            # Recompensa base por velocidad
                            reward = car.speed / car.max_speed
                            
                            # NUEVO: Bonus por sobrevivir más tiempo
                            reward += 0.5
                            
                            # Penalizar proximidad a bordes
                            min_sensor = min(sensor_distances)
                            if min_sensor < car.max_sensor_distance * 0.2:
                                reward -= 0.3
                            
                            # NUEVO: Bonus por mantener distancia segura
                            elif min_sensor > car.max_sensor_distance * 0.5:
                                reward += 0.2
                        
                        agent.total_reward += reward
                        agent.steps += 1
                        # MEJORADO: Fitness que valora más la supervivencia
                        agent.fitness = agent.total_reward + agent.steps * 0.5
                        
                        # Actualizar posición para renderizado
                        agent.x = car.x
                        agent.y = car.y
                        agent.angle = car.angle
                        agent.speed = car.speed
                    
                    # Si todos murieron, nueva generación
                    if all_done:
                        self.select_best_and_evolve()
                        self.reset_generation()
                        break
            
            # Renderizar
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
        print("\n✓ Entrenamiento finalizado")
    
    def check_collision(self, car):
        """Verifica colisión del auto"""
        px, py = int(car.x), int(car.y)
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return True
        if self.env.track.get_at((px, py)) == (0, 0, 0):
            return True
        return False
    
    def render(self):
        """Renderiza el entorno y los agentes"""
        # Dibujar pista
        self.screen.blit(self.env.track, (0, 0))
        
        # Dibujar agentes
        for agent in self.agents:
            if agent.is_alive:
                # Dibujar auto
                car_surf = pygame.Surface((20, 40))
                car_surf.fill(agent.color)
                rotated_surf = pygame.transform.rotate(car_surf, -np.degrees(agent.angle))
                rotated_rect = rotated_surf.get_rect(center=(int(agent.x), int(agent.y)))
                self.screen.blit(rotated_surf, rotated_rect.topleft)
                
                # Etiqueta con fitness
                font = pygame.font.Font(None, 16)
                label = font.render(f"#{agent.agent_id}: {agent.fitness:.0f}", True, WHITE)
                self.screen.blit(label, (int(agent.x) - 20, int(agent.y) - 30))
        
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


if __name__ == "__main__":
    trainer = GeneticTrainer(width=1920, height=1080)
    trainer.run()
