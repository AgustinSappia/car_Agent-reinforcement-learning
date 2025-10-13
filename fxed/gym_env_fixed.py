import gymnasium as gym
import numpy as np
import pygame
from environment_fixed import Environment

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
        
        # CORREGIDO: Simplificado el cálculo de resolución dinámica
        if self.dynamic_resolution and self.render_mode == 'human':
            # Obtener info de display sin inicializar completamente
            import pygame.display
            info = pygame.display.Info()
            nat_w, nat_h = info.current_w, info.current_h
            
            # Calcular escala manteniendo aspect ratio, sin exceder resolución nativa
            scale_w = nat_w / self._target_width
            scale_h = nat_h / self._target_height
            scale = min(scale_w, scale_h, 1.0)  # Máximo 1.0 (no upscale)
            
            # Aplicar escala con mínimo de 1280x720
            self._width = max(int(self._target_width * scale), 1280)
            self._height = max(int(self._target_height * scale), 720)
            
            # Ajustar para mantener aspect ratio 16:9
            aspect_ratio = 16 / 9
            current_aspect = self._width / self._height
            if abs(current_aspect - aspect_ratio) > 0.05:
                self._height = int(self._width / aspect_ratio)
        
        # Asegurar enteros para PyGame
        self._width = int(self._width)
        self._height = int(self._height)
        
        self.env = Environment(width=self._width, height=self._height)
        
        # CORREGIDO: Observación normalizada [0, 1] para todas las dimensiones
        # 5 sensores + velocidad + ángulo → 7 dims, todas normalizadas
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]*7, dtype=np.float32),
            high=np.array([1.0]*7, dtype=np.float32),
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
        """
        CORREGIDO: Normalización consistente de observación.
        """
        super().reset(seed=seed)
        
        obs = self.env.reset()  # Devuelve sensores normalizados (5 valores)
        
        # Agregar velocidad y ángulo normalizados
        speed_norm = self.env.car.speed / self.env.car.max_speed
        angle_norm = (self.env.car.angle + np.pi) / (2 * np.pi)  # Normalizar [-π, π] a [0, 1]
        
        obs = np.array(list(obs) + [speed_norm, angle_norm], dtype=np.float32)
        
        # Inicializar PyGame solo si es necesario
        if self.render_mode == 'human' and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"DQN Training - Self Driving Car (Res: {self.width}x{self.height})")
            self.clock = pygame.time.Clock()
        
        return obs, {}

    def step(self, action):
        """
        CORREGIDO: Normalización consistente y manejo de info.
        """
        # action es [steering_idx (0-2), throttle_idx (0-1)] → mapear a [-1,0,1], [0,1]
        steering = action[0] - 1  # 0→-1, 1→0, 2→1
        throttle = action[1]      # 0 o 1
        
        obs, reward, done, info = self.env.step([steering, throttle])
        
        # Agregar velocidad y ángulo normalizados
        speed_norm = info['speed'] / self.env.car.max_speed
        angle_norm = (info['angle'] + np.pi) / (2 * np.pi)  # Normalizar [-π, π] a [0, 1]
        
        obs = np.array(list(obs) + [speed_norm, angle_norm], dtype=np.float32)
        
        truncated = False
        
        # Renderizar si está en modo human
        if self.render_mode == 'human':
            self.env.render(self.screen)
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)
        
        return obs, reward, done, truncated, info

    def render(self):
        """Renderiza el entorno (ya se hace en step si render_mode='human')."""
        if self.render_mode == 'human' and self.screen:
            self.env.render(self.screen)
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)

    def close(self):
        """Cierra el entorno y PyGame."""
        if self.screen:
            pygame.quit()
            self.screen = None
            self.clock = None
