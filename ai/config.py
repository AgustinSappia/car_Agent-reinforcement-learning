"""
Configuración del agente: qué ve (sensores), cómo piensa (red neuronal),
cómo maneja (acciones y física) y cómo evoluciona (algoritmo genético).

Se guarda como JSON en la carpeta agentes/ para poder tener varios "perfiles".
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict

PROFILES_DIR = 'agentes'

# Conjuntos de acciones: (giro, acelerar, frenar)
ACTION_SETS = {
    'simple': {
        'label': "Simple (3): izq / recto / der, siempre acelera",
        'actions': [(-1, 1, 0), (0, 1, 0), (1, 1, 0)],
    },
    'clasico': {
        'label': "Clásico (4): nada / acelerar / izq / der",
        'actions': [(0, 0, 0), (0, 1, 0), (-1, 1, 0), (1, 1, 0)],
    },
    'completo': {
        'label': "Completo (7): también frena y gira sin acelerar",
        'actions': [(0, 0, 0), (0, 1, 0), (-1, 1, 0), (1, 1, 0), (0, 0, 1), (-1, 0, 0), (1, 0, 0)],
    },
}

# Tamaños de red: capas ocultas
BRAIN_SIZES = {
    'mini': [6],
    'chico': [10],
    'mediano': [16, 8],
    'grande': [32, 16],
    'enorme': [64, 32],
}


@dataclass
class AgentConfig:
    # Sensores
    num_sensors: int = 5
    sensor_spread: int = 120       # grados entre el primer y el último sensor
    sensor_range: int = 220        # píxeles
    use_speed: bool = True         # la velocidad propia como entrada
    use_compass: bool = True       # dirección hacia el próximo objetivo como entrada

    # Cerebro
    brain_size: str = 'mediano'

    # Manejo
    action_set: str = 'clasico'
    max_speed: float = 5.0
    turn_speed: float = 0.1        # radianes por paso

    # Evolución
    population: int = 40
    elite_pct: int = 20
    mutation_rate: float = 0.1     # probabilidad de mutar cada peso
    mutation_strength: float = 0.3
    crossover: bool = True
    max_steps: int = 2500          # pasos por generación
    patience: int = 250            # pasos sin progresar antes de descartar al auto

    # ---------------------------------------------------------------- #
    def sensor_angles(self):
        """Ángulos relativos al frente del auto, en radianes"""
        n = self.num_sensors
        if n == 1:
            return [0.0]
        spread = math.radians(self.sensor_spread)
        return [-spread / 2 + spread * i / (n - 1) for i in range(n)]

    def input_size(self):
        return self.num_sensors + (1 if self.use_speed else 0) + (2 if self.use_compass else 0)

    def actions(self):
        return ACTION_SETS[self.action_set]['actions']

    def layer_sizes(self):
        return [self.input_size()] + BRAIN_SIZES[self.brain_size] + [len(self.actions())]

    def num_params(self):
        sizes = self.layer_sizes()
        return sum(a * b + b for a, b in zip(sizes, sizes[1:]))

    def brain_signature(self):
        """Dos configuraciones con la misma firma pueden compartir cerebro"""
        return 'x'.join(str(s) for s in self.layer_sizes())

    # ---------------------------------------------------------------- #
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, name):
        os.makedirs(PROFILES_DIR, exist_ok=True)
        with open(os.path.join(PROFILES_DIR, f"{name}.json"), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, name):
        with open(os.path.join(PROFILES_DIR, f"{name}.json"), 'r') as f:
            return cls.from_dict(json.load(f))


def list_profiles():
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith('.json'))


def brain_path(profile, scenario):
    return os.path.join(PROFILES_DIR, f"{profile}_{scenario}.npz")


PRESETS = {
    "Equilibrado": AgentConfig(),
    "Rápido de entrenar": AgentConfig(num_sensors=3, sensor_spread=90, brain_size='chico',
                                      action_set='simple', population=30, max_steps=1800),
    "Explorador": AgentConfig(population=80, mutation_rate=0.2, mutation_strength=0.5,
                              elite_pct=10, patience=400),
    "Preciso": AgentConfig(num_sensors=9, sensor_spread=180, sensor_range=300, brain_size='grande',
                           action_set='completo', population=60, max_steps=4000),
}
