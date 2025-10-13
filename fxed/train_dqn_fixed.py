import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
from gym_env_fixed import CarEnv
from model import LinearQNet
import pygame
import os
import csv

class DQNAgent:
    """
    Agente DQN que aprende a conducir el auto usando Deep Q-Learning.
    - Usa dos redes: policy_net (actualiza en cada paso) y target_net (estable para targets).
    - Experience Replay: almacena transiciones (s, a, r, s', done) en un buffer.
    - Epsilon-greedy para balancear exploración/explotación.
    - Entrenamiento: MSE loss entre Q predicho y target Q (Bellman equation).
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 batch_size=64, memory_size=100_000, update_target_every=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.update_target_every = update_target_every
        self.steps = 0

        # Redes Q
        self.policy_net = LinearQNet(state_size, output_size=action_size)
        self.target_net = LinearQNet(state_size, output_size=action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizador y loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        print(f"Usando dispositivo: {self.device}")

    def remember(self, state, action, reward, next_state, done):
        """Almacena una transición en el buffer de memoria."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Selecciona acción usando epsilon-greedy."""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return np.argmax(q_values.cpu().data.numpy()).item()

    def act_greedy(self, state):
        """NUEVO: Selecciona acción sin exploración (solo explotación)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return np.argmax(q_values.cpu().data.numpy()).item()

    def train_step(self):
        """Entrena la red policy_net con un batch del buffer."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Q-values actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-values target
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Loss y actualización
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Actualizar target_net
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network actualizada en paso {self.steps}")
        
        return loss.item()

    def save(self, file_name):
        """Guarda el modelo (policy_net y optimizer)."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        torch.save(checkpoint, file_name)
        print(f"Modelo guardado: {file_name}")

    def load(self, file_name):
        """NUEVO: Carga modelo pre-entrenado."""
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            print(f"Modelo cargado: {file_name} (Epsilon: {self.epsilon:.3f}, Steps: {self.steps})")
            return True
        return False


# NUEVO: Configuración centralizada
CONFIG = {
    'state_size': 7,
    'action_size': 4,
    'episodes': 2000,
    'max_steps': 1000,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_min': 0.1,
    'epsilon_decay': 0.995,
    'batch_size': 64,
    'memory_size': 100_000,
    'update_target_every': 1000,
    'save_every': 100,
    'model_path': 'dqn_car_model.pth',
    'best_model_path': 'dqn_car_model_best.pth',
    'log_file': 'training_log.csv'
}

# Mapping de acciones
ACTION_MAP = [
    [1, 0],  # 0: nada
    [1, 1],  # 1: acelerar
    [0, 1],  # 2: izquierda + acel
    [2, 1]   # 3: derecha + acel
]

# Crear entorno
env = CarEnv(render_mode='human', dynamic_resolution=True)
print(f"Resolución: {env.width}x{env.height} | Sensor max: {env.env.car.max_sensor_distance}")

# Crear agente
agent = DQNAgent(
    state_size=CONFIG['state_size'],
    action_size=CONFIG['action_size'],
    learning_rate=CONFIG['learning_rate'],
    gamma=CONFIG['gamma'],
    epsilon=CONFIG['epsilon_start'],
    epsilon_min=CONFIG['epsilon_min'],
    epsilon_decay=CONFIG['epsilon_decay'],
    batch_size=CONFIG['batch_size'],
    memory_size=CONFIG['memory_size'],
    update_target_every=CONFIG['update_target_every']
)

# NUEVO: Cargar modelo si existe
agent.load(CONFIG['model_path'])

# Métricas
episode_rewards = []
avg_reward_window = deque(maxlen=10)
best_avg_reward = -float('inf')

# NUEVO: Inicializar archivo de log
with open(CONFIG['log_file'], 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Episode', 'Reward', 'Avg_Reward', 'Epsilon', 'Steps', 'Loss'])

pygame.init()

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO DQN")
print("="*60)
print(f"Resolución: {env.width}x{env.height}")
print(f"Estado: {CONFIG['state_size']} dims | Acciones: {CONFIG['action_size']}")
print(f"Episodios: {CONFIG['episodes']} | Max steps: {CONFIG['max_steps']}")
print(f"Dispositivo: {agent.device}")
print("="*60 + "\n")

try:
    for episode in range(CONFIG['episodes']):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        losses = []

        while not done and step_count < CONFIG['max_steps']:
            # Seleccionar acción
            action_idx = agent.act(state)
            mapped_action = ACTION_MAP[action_idx]

            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(mapped_action)
            done = done or truncated
            total_reward += reward

            # Almacenar experiencia
            agent.remember(state, action_idx, reward, next_state, done)

            # Entrenar
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            step_count += 1

            # Renderizar con info
            if hasattr(env, 'screen') and env.screen:
                # Área de texto
                text_area_width = int(env.width * 0.35)
                text_area_height = int(env.height * 0.08)
                pygame.draw.rect(env.screen, (0, 0, 0), (0, 0, text_area_width, text_area_height))
                
                font_size = max(int(env.height * 0.02), 16)
                font = pygame.font.Font(None, font_size)
                
                pos_x = int(env.width * 0.01)
                pos_y = int(env.height * 0.01)
                
                text_episode = font.render(f"Episodio: {episode + 1}/{CONFIG['episodes']}", True, (255, 255, 255))
                text_reward = font.render(f"Reward: {total_reward:.1f} | ε: {agent.epsilon:.3f} | Steps: {step_count}", True, (255, 255, 255))
                
                env.screen.blit(text_episode, (pos_x, pos_y))
                env.screen.blit(text_reward, (pos_x, pos_y + font_size))
                
                pygame.display.flip()

        # Fin del episodio
        episode_rewards.append(total_reward)
        avg_reward_window.append(total_reward)
        avg_reward = np.mean(avg_reward_window)
        avg_loss = np.mean(losses) if losses else 0

        # NUEVO: Guardar mejor modelo
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save(CONFIG['best_model_path'])
            print(f"✓ Nuevo mejor modelo! Avg reward: {avg_reward:.1f}")

        # NUEVO: Logging a CSV
        with open(CONFIG['log_file'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward, avg_reward, agent.epsilon, step_count, avg_loss])

        # Imprimir progreso
        if (episode + 1) % 10 == 0:
            print(f"Ep {episode + 1}/{CONFIG['episodes']} | Reward: {total_reward:.1f} | "
                  f"Avg(10): {avg_reward:.1f} | ε: {agent.epsilon:.3f} | Steps: {step_count} | Loss: {avg_loss:.4f}")

        # Guardar modelo periódicamente
        if (episode + 1) % CONFIG['save_every'] == 0:
            agent.save(CONFIG['model_path'])

except KeyboardInterrupt:
    print("\n\nEntrenamiento interrumpido por el usuario.")
    agent.save(CONFIG['model_path'])
except Exception as e:
    print(f"\n\nError durante entrenamiento: {e}")
    import traceback
    traceback.print_exc()
    agent.save(CONFIG['model_path'])
finally:
    # Guardar modelo final
    print("\n" + "="*60)
    print("ENTRENAMIENTO FINALIZADO")
    print("="*60)
    if len(episode_rewards) >= 100:
        print(f"Recompensa promedio (últimos 100 ep): {np.mean(episode_rewards[-100:]):.1f}")
    else:
        print(f"Recompensa promedio: {np.mean(episode_rewards):.1f}")
    print(f"Mejor recompensa promedio: {best_avg_reward:.1f}")
    print(f"Modelo guardado en: {CONFIG['model_path']}")
    print(f"Mejor modelo en: {CONFIG['best_model_path']}")
    print(f"Log guardado en: {CONFIG['log_file']}")
    print("="*60 + "\n")
    
    agent.save(CONFIG['model_path'])
    env.close()
    pygame.quit()
