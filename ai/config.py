"""
Configuración del agente: qué ve (sensores), cómo piensa (red neuronal),
cómo maneja (acciones y física), cómo evoluciona (algoritmo genético) y
qué premia su puntaje.

Hay un tipo de agente por escenario ('pista', 'laberinto', 'futbol'). Cada tipo
tiene sus propias opciones, presets y perfiles: agentes/<tipo>/<nombre>.json
"""

import colorsys
import json
import math
import os
import random
from dataclasses import dataclass, field, asdict, replace

PROFILES_DIR = 'agentes'
KINDS = ('pista', 'laberinto', 'futbol')
KIND_LABELS = {'pista': "Pista", 'laberinto': "Laberinto", 'futbol': "Fútbol"}

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


def random_color(avoid=()):
    """Color vivo al azar para reconocer a cada agente. Si se pasan colores en avoid,
    elige el tono más distinto a todos ellos (entre varios al azar)."""
    used = [colorsys.rgb_to_hsv(*(c / 255 for c in col))[0] for col in avoid]

    def distance(h):
        return min((min(abs(h - u), 1 - abs(h - u)) for u in used), default=1)
    h = max((random.random() for _ in range(12)), key=distance)
    r, g, b = colorsys.hsv_to_rgb(h, random.uniform(0.55, 0.85), random.uniform(0.85, 1.0))
    return [int(r * 255), int(g * 255), int(b * 255)]


def used_colors():
    """Colores de los perfiles guardados (de todos los tipos)"""
    out = []
    for kind in KINDS:
        for name in list_profiles(kind):
            try:
                with open(profile_path(kind, name)) as f:
                    out.append(json.load(f)['color'])
            except Exception:
                pass
    return out


@dataclass
class AgentConfig:
    kind: str = 'pista'
    color: list = field(default_factory=random_color)

    # Sensores
    num_sensors: int = 5
    sensor_spread: int = 120       # grados entre el primer y el último sensor
    sensor_range: int = 220        # píxeles
    use_speed: bool = True         # la velocidad propia como entrada
    use_compass: bool = True       # pista/laberinto: dirección al próximo objetivo
    see_mates: bool = True         # fútbol: compañero más cercano
    see_rivals: bool = True        # fútbol: rival más cercano
    use_role: bool = True          # fútbol: número de jugador dentro del equipo

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
    max_steps: int = 2500          # pista/laberinto: pasos por generación
    patience: int = 250            # pista/laberinto: pasos sin progresar antes de descartar al auto

    # Puntaje de pista y laberinto
    r_fast: float = 2.0            # premio por cada paso que sobra al llegar
    p_crash: int = 0               # castigo por chocar

    # Puntaje de fútbol
    r_goal: int = 1000
    p_conceded: int = 600
    p_own_goal: int = 800          # además del gol en contra
    r_touch: int = 15
    r_advance: float = 1.0         # por cada píxel que acerca la pelota al arco rival
    r_near: int = 200              # estar cerca de la pelota
    p_crowd: int = 300             # amontonarse con un compañero
    p_idle: int = 300              # quedarse quieto

    # ---------------------------------------------------------------- #
    def sensor_angles(self):
        """Ángulos relativos al frente del auto, en radianes"""
        n = self.num_sensors
        if n == 1:
            return [0.0]
        spread = math.radians(self.sensor_spread)
        return [-spread / 2 + spread * i / (n - 1) for i in range(n)]

    def football_inputs(self):
        # pelota (3) + su velocidad (2) + arco rival (3) + arco propio (2) + patada lista (1)
        return 11 + (3 if self.see_mates else 0) + (3 if self.see_rivals else 0) + (1 if self.use_role else 0)

    def input_size(self, kind=None):
        kind = kind or self.kind
        base = self.num_sensors + (1 if self.use_speed else 0)
        if kind == 'futbol':
            return base + self.football_inputs()
        return base + (2 if self.use_compass else 0)

    def actions(self, kind=None):
        """Lista de (giro, acelerar, frenar, patear)"""
        kind = kind or self.kind
        acts = [tuple(a) + (0,) for a in ACTION_SETS[self.action_set]['actions']]
        if kind == 'futbol':
            acts.append((0, 1, 0, 1))
        return acts

    def layer_sizes(self, kind=None):
        return [self.input_size(kind)] + BRAIN_SIZES[self.brain_size] + [len(self.actions(kind))]

    def num_params(self, kind=None):
        sizes = self.layer_sizes(kind)
        return sum(a * b + b for a, b in zip(sizes, sizes[1:]))

    # ---------------------------------------------------------------- #
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, name):
        os.makedirs(os.path.join(PROFILES_DIR, self.kind), exist_ok=True)
        with open(profile_path(self.kind, name), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, kind, name):
        with open(profile_path(kind, name), 'r') as f:
            cfg = cls.from_dict(json.load(f))
        cfg.kind = kind
        return cfg


def profile_path(kind, name):
    return os.path.join(PROFILES_DIR, kind, f"{name}.json")


def list_profiles(kind):
    folder = os.path.join(PROFILES_DIR, kind)
    if not os.path.isdir(folder):
        return []
    return sorted(f[:-5] for f in os.listdir(folder) if f.endswith('.json'))


def brain_path(profile, kind, suffix=''):
    return os.path.join(PROFILES_DIR, kind, f"{profile}{suffix}.npz")


# ---------------------------------------------------------------------- #
# Presets por tipo de agente
# ---------------------------------------------------------------------- #
PRESETS = {
    'pista': {
        "Equilibrado": dict(),
        "Rápido de entrenar": dict(num_sensors=3, sensor_spread=90, brain_size='chico', action_set='simple',
                                   population=30, max_steps=1800),
        "Explorador": dict(population=80, mutation_rate=0.2, mutation_strength=0.5, elite_pct=10, patience=400),
        "Preciso": dict(num_sensors=9, sensor_spread=180, sensor_range=300, brain_size='grande',
                        action_set='completo', population=60, max_steps=4000),
    },
    'laberinto': {
        "Explorador": dict(num_sensors=7, sensor_spread=180, sensor_range=300, turn_speed=0.15, max_speed=4.0,
                           patience=400, max_steps=3000),
        "Rápido de entrenar": dict(num_sensors=5, sensor_spread=180, brain_size='chico', action_set='simple',
                                   turn_speed=0.15, max_speed=4.0),
        "Generalista": dict(num_sensors=9, sensor_spread=270, sensor_range=300, brain_size='grande',
                            turn_speed=0.15, max_speed=4.0, population=80, patience=500, max_steps=4000),
        "Sin brújula": dict(num_sensors=7, sensor_spread=180, sensor_range=300, turn_speed=0.15, max_speed=4.0,
                            use_compass=False, patience=400, max_steps=3000),
    },
    'futbol': {
        "Goleador": dict(num_sensors=5, sensor_spread=180),
        "Equipo ordenado": dict(num_sensors=5, sensor_spread=180, p_crowd=700, r_near=120),
        "Defensor": dict(num_sensors=5, sensor_spread=180, p_conceded=1200, p_own_goal=1500, r_advance=0.6),
        "Rápido de entrenar": dict(num_sensors=3, sensor_spread=180, brain_size='chico', see_rivals=False,
                                   population=40),
    },
}


def preset(kind, name, color=None):
    cfg = AgentConfig(kind=kind, **PRESETS[kind][name])
    if color is not None:
        cfg.color = list(color)
    return cfg


def default_config(kind):
    """Configuración inicial de un agente nuevo: el primer preset de su tipo, con un color distinto a los ya usados"""
    return preset(kind, next(iter(PRESETS[kind])), color=random_color(used_colors()))


def copy_config(cfg):
    return replace(cfg, color=list(cfg.color))
