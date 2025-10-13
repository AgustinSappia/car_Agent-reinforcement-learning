import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym  # Para compatibilidad con wrappers Gym
from gym_env import CarEnv  # Wrapper Gym del entorno existente
from model import LinearQNet
import pygame  # Para manejo de pantalla y texto
import os

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
        self.gamma = gamma  # Factor de descuento para recompensas futuras
        self.epsilon = epsilon  # Probabilidad inicial de acción aleatoria (exploración)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)  # Buffer de experiencia
        self.update_target_every = update_target_every
        self.steps = 0  # Contador de pasos para actualizar target_net

        # Redes Q
        self.policy_net = LinearQNet(state_size, output_size=action_size)
        self.target_net = LinearQNet(state_size, output_size=action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Inicializar target con policy

        # Optimizador y loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Dispositivo (CPU por defecto; usa 'cuda' si tienes GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def remember(self, state, action, reward, next_state, done):
        """
        Almacena una transición en el buffer de memoria.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selecciona acción usando epsilon-greedy.
        - Con probabilidad epsilon: acción aleatoria (exploración).
        - Sino: acción greedy (argmax de Q-values de policy_net).
        """
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploración: aleatorio

        # Explotación: usar red policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_tensor)
        return np.argmax(q_values.cpu().data.numpy()).item()

    def train_step(self):
        """
        Entrena la red policy_net con un batch del buffer.
        - Si buffer < batch_size, no entrena.
        - Calcula targets usando target_net (Bellman: r + gamma * max Q(s', a')).
        - Actualiza policy_net minimizando MSE.
        - Decae epsilon gradualmente.
        """
        if len(self.memory) < self.batch_size:
            return

        # Samplear batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Q-values actuales (policy_net)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-values target (target_net)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Loss y actualización
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Actualizar target_net cada N pasos
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network actualizada en paso {self.steps}")

    def save(self, file_name):
        """
        Guarda el modelo (policy_net y optimizer).
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, file_name)

# Configuración de entrenamiento
STATE_SIZE = 7  # 5 sensores + speed + angle (de gym_env.py)
ACTION_SIZE = 4  # Acciones discretas (mapeadas a MultiDiscrete)
EPISODES = 2000  # Número de episodios (ajustable)
MAX_STEPS_PER_EPISODE = 1000  # Límite de pasos por episodio para evitar loops
SAVE_EVERY = 100  # Guardar modelo cada N episodios
MODEL_PATH = 'dqn_car_model.pth'

# Mapping de acciones DQN (int 0-3) a formato MultiDiscrete del env [steering_idx (0-2), throttle_idx (0-1)]
ACTION_MAP = [
    [1, 0],  # 0: nada (steering=0, throttle=0)
    [1, 1],  # 1: acelerar (steering=0, throttle=1)
    [0, 1],  # 2: izquierda + acel (steering=-1, throttle=1)
    [2, 1]   # 3: derecha + acel (steering=1, throttle=1)
]

# Crear entorno Gym wrapper con render_mode='human' y resolución dinámica
env = CarEnv(render_mode='human', dynamic_resolution=True)  # True: adapta a monitor; False: fija 1920x1080
print(f"Resolución final: {env.width}x{env.height} | Sensor max: {env.env.car.max_sensor_distance}")
# Crear agente DQN
agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

# Métricas
episode_rewards = []
avg_reward_window = deque(maxlen=10)  # Ventana móvil de 10 episodios para promedio

# Inicializar PyGame si no está hecho en CarEnv (por seguridad)
pygame.init()

print("Iniciando entrenamiento DQN...")
print(f"Resolución: {env.width}x{env.height} (dinámica)")
print(f"Estado: {STATE_SIZE} dims | Acciones: {ACTION_SIZE}")
print("El auto aprenderá a conducir: observa cómo mejora con los episodios.")

for episode in range(EPISODES):
    state, _ = env.reset()  # Reset del entorno (devuelve state y info)
    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS_PER_EPISODE:
        # Seleccionar acción (int 0-3)
        action_idx = agent.act(state)

        # Mapear a formato del env: [steering_idx, throttle_idx]
        mapped_action = ACTION_MAP[action_idx]

        # Ejecutar acción en el entorno
        next_state, reward, done, truncated, info = env.step(mapped_action)
        done = done or truncated  # Gym usa truncated para timeouts
        total_reward += reward

        # Almacenar experiencia (usa action_idx original para el agente)
        agent.remember(state, action_idx, reward, next_state, done)

        # Entrenar un paso
        agent.train_step()

        # Actualizar estado
        state = next_state
        step_count += 1

        # Renderizar en tiempo real (ya se hace en env.step(), pero agregamos texto aquí)
        # Blit texto en pantalla después del render base (escalado dinámicamente)
        if hasattr(env, 'screen') and env.screen:
            # Limpiar área de texto (fondo negro, tamaño proporcional)
            text_area_width = env.width * 0.3
            text_area_height = env.height * 0.06
            pygame.draw.rect(env.screen, (0, 0, 0), (0, 0, text_area_width, text_area_height))
            
            # Fuente escalada proporcionalmente (2% de height)
            font_size = int(env.height * 0.02)
            font = pygame.font.Font(None, font_size)
            
            # Posición proporcional (1% de width/height)
            pos_x = env.width * 0.01
            pos_y = env.height * 0.01
            
            # Texto: episodio y recompensa actual
            text_episode = font.render(f"Episodio: {episode + 1}/{EPISODES}", True, (255, 255, 255))
            text_reward = font.render(f"Recompensa: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f} | Pasos: {step_count}", True, (255, 255, 255))
            env.screen.blit(text_episode, (pos_x, pos_y))
            env.screen.blit(text_reward, (pos_x, pos_y + font_size))
            
            # Actualizar pantalla
            pygame.display.flip()

    # Fin del episodio
    episode_rewards.append(total_reward)
    avg_reward_window.append(total_reward)
    avg_reward = np.mean(avg_reward_window)

    # Decaimiento de epsilon por episodio (adicional al de train_step)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    # Imprimir progreso cada 10 episodios
    if (episode + 1) % 10 == 0:
        print(f"Episodio {episode + 1}/{EPISODES} | Recompensa: {total_reward:.1f} | Promedio (últ 10): {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f} | Pasos: {step_count}")

    # Guardar modelo cada SAVE_EVERY episodios
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save(MODEL_PATH)
        print(f"Modelo guardado en episodio {episode + 1}")

# Entrenamiento completado
print("Entrenamiento finalizado!")
if len(episode_rewards) >= 100:
    print(f"Recompensa promedio final (últ 100 episodios): {np.mean(episode_rewards[-100:]):.1f}")
else:
    print(f"Recompensa promedio final: {np.mean(episode_rewards):.1f}")
agent.save(MODEL_PATH)  # Guardar modelo final

# Cerrar entorno
env.close()
pygame.quit()