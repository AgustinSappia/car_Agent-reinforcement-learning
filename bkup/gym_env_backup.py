import gymnasium as gym
import numpy as np
import pygame
from environment import Environment

class CarEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, width=1920, height=1080, dynamic_resolution=True):
        super().__init__()
        self.render_mode = render_mode
        self.dynamic_resolution = dynamic_resolution
        self._target_width = width
        self._target_height = height
        self.screen = None
        self.clock = None
        
        # Detectar resolución dinámica si se habilita
        self._width = self._target_width
        self._height = self._target_height
        if self.dynamic_resolution and self.render_mode == 'human':
            pygame.init()  # Inicializar temporalmente para Info()
            info = pygame.display.Info()
            nat_w, nat_h = info.current_w, info.current_h
            pygame.quit()  # Cerrar temporal
            
            # Calcular scale manteniendo 16:9 (1920:1080 = 16:9)
            aspect_ratio = 16 / 9
            target_aspect = self._target_width / self._target_height  # ~1.777
            nat_aspect = nat_w / nat_h
            
            # Scale basado en el menor lado, pero no exceder nativa
            scale_w = nat_w / self._target_width
            scale_h = nat_h / self._target_height
            scale = min(scale_w, scale_h, 1.0)  # Máx 1.0 (no upscale)
            
            # Mínimo: 1280x720 para usabilidad; si nativa < eso, usa 80% de nativa ajustada a 16:9
            min_w, min_h = 1280, 720
            if nat_w < min_w or nat_h < min_h:
                scale = min(0.8, scale)
                # Ajustar height para 16:9 si nativa no lo es
                adjusted_h = int(nat_w * 0.8 / aspect_ratio)
                self._width = int(nat_w * 0.8)
                self._height = min(adjusted_h, int(nat_h * 0.8))  # No exceder nat_h
            else:
                self._width = int(self._target_width * scale)
                self._height = int(self._target_height * scale)
                # Corregir si aspect ratio se distorsiona
                if abs((self._width / self._height) - aspect_ratio) > 0.05:
                    self._height = int(self._width / aspect_ratio)
        
        # Asegurar enteros para PyGame
        self._width = int(self._width)
        self._height = int(self._height)
        
        self.env = Environment(width=self._width, height=self._height)
        
        # Observación: 5 sensores + velocidad + ángulo → 7 dims
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*5 + [0, 0], dtype=np.float32),
            high=np.array([self.env.car.max_sensor_distance]*5 + [self.env.car.max_speed, 2*np.pi], dtype=np.float32),
            dtype=np.float32
        )
        # Acción: [steering_idx (0-2), throttle_idx (0-1)] → MultiDiscrete [3,2]
        self.action_space = gym.spaces.MultiDiscrete([3, 2])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        obs = np.array(list(obs) + [self.env.car.speed, self.env.car.angle], dtype=np.float32)
        if self.render_mode == 'human' and self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"DQN Training - Self Driving Car (Res: {self.width}x{self.height})")
            self.clock = pygame.time.Clock()
            pygame.init()  # Re-inicializar para eventos y display
        return obs, {}

    def step(self, action):
        # action es [steering_idx (0-2), throttle_idx (0-1)] → mapear a [-1,0,1], [0,1]
        steering = action[0] - 1  # 0→-1, 1→0, 2→1
        throttle = action[1]      # 0 o 1
        obs, reward, done, info = self.env.step([steering, throttle])
        obs = np.array(list(obs) + [info['speed'], info['angle']], dtype=np.float32)
        truncated = False
        if self.render_mode == 'human':
            self.env.render(self.screen)
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)
        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode == 'human' and self.screen:
            self.env.render(self.screen)
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()