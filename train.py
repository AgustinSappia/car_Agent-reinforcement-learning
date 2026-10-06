"""
Autos que aprenden solos: entrenamiento con algoritmo genético y modos para mostrarlos.

Flujo:  menú principal
        - Entrenar: pista / varias pistas / laberinto / fútbol -> taller del agente -> entrenamiento
        - Campeones: modo Expo (generación 1 contra el campeón) o jugar contra la IA

Uso:  python train.py
"""

import os

import pygame

from ai import ui
from ai.config import AgentConfig, PROFILES_DIR, default_config
from ai.football import Field, FootballScenario
from ai.menus import (main_menu, maze_settings, workshop, loading, curriculum_settings,
                      football_settings, gallery)
from ai.scenarios import PistaScenario, LaberintoScenario, CurriculumScenario

CAPTION = "Autos que aprenden - Entrenamiento genético"


def _last_file(kind):
    return os.path.join(PROFILES_DIR, kind, '_ultimo.txt')


def last_profile(kind):
    """Último perfil usado de ese tipo de agente (o uno nuevo con el primer preset)"""
    try:
        with open(_last_file(kind)) as f:
            name = f.read().strip()
        return name, AgentConfig.load(kind, name)
    except Exception:
        return 'perfil', default_config(kind)


def remember_profile(kind, name):
    os.makedirs(os.path.join(PROFILES_DIR, kind), exist_ok=True)
    with open(_last_file(kind), 'w') as f:
        f.write(name)


def load_map(name, title=None):
    """Datos de una pista o cancha guardada, con el nombre visible y el de archivo"""
    from track_loader import load_track_data
    data = load_track_data(name)
    if data:
        data['file'] = name
        from track_selector import display_name
        data['name'] = title or display_name(data.get('metadata') or {}, name)
    return data


def pick_from_selector(kind):
    """Abre el selector (su propia ventana). Devuelve (pantalla, datos o None)."""
    from track_selector import select_track
    pygame.quit()
    selected = select_track(kind=kind)
    screen = ui.open_window(CAPTION)
    if not selected:
        return screen, None
    loading(screen, "Cargando...")
    return screen, load_map(selected['name'], selected.get('title'))


# ---------------------------------------------------------------------- #
# Preparar cada escenario
# ---------------------------------------------------------------------- #
def setup_pista(screen):
    screen, data = pick_from_selector('pista')
    if not data:
        return screen, None
    loading(screen, "Preparando la pista...")
    return screen, PistaScenario(data)


def setup_curriculum(screen, opts):
    names = curriculum_settings(screen, opts)
    if names in (None, 'quit'):
        return screen, names
    scenarios = []
    for i, name in enumerate(names):
        loading(screen, f"Preparando pista {i + 1} de {len(names)}...")
        data = load_map(name)
        if data:
            scenarios.append(PistaScenario(data))
    if not scenarios:
        return screen, None
    return screen, CurriculumScenario(scenarios, opts['rule'], opts['every'], opts['threshold'])


def setup_football(screen, opts):
    field = opts.get('field') or Field()
    while True:
        action, scenario = football_settings(screen, opts, field)
        if action == 'pick':
            screen, data = pick_from_selector('cancha')
            if data and data.get('goals'):
                field = Field(data)
        elif action == 'classic':
            field = Field()
        else:
            opts['field'] = field
            return screen, (scenario if action == 'go' else ('quit' if action == 'quit' else None))


def scenario_for_entry(screen, entry, other_map=False):
    """Arma el escenario en el que se entrenó un cerebro de la galería"""
    meta, kind = entry['meta'], entry['kind']
    loading(screen, "Preparando...")
    if kind == 'laberinto':
        maze = meta.get('maze') or {}
        return screen, LaberintoScenario(maze.get('size', 'chico'), maze.get('new_every', 0),
                                         maze.get('braid', False), maze_seed=maze.get('seed'))
    if kind == 'futbol':
        field = Field()
        if meta.get('map_file'):
            data = load_map(meta['map_file'])
            if data and data.get('goals'):
                field = Field(data)
        return screen, FootballScenario(field, meta.get('team_size', 1), meta.get('opponent', 'bot_normal'),
                                        meta.get('match_steps', 1800), meta.get('ball_control', 'libre'),
                                        meta.get('random_starts', 0.0))
    data = None if other_map else load_map(meta.get('map_file', ''))
    if not data:
        screen, data = pick_from_selector('pista')
        if not data:
            return screen, None
    loading(screen, "Preparando la pista...")
    return screen, PistaScenario(data)


# ---------------------------------------------------------------------- #
def train_loop(screen, scenario):
    """Taller <-> entrenamiento. Devuelve True si hay que salir del programa."""
    from ai.trainer import Trainer
    from ai.football_trainer import FootballTrainer
    profile, cfg = last_profile(scenario.key)
    while True:
        action, cfg, profile = workshop(screen, scenario, cfg, profile)
        if action == 'quit':
            return True
        if action == 'back':
            return False
        remember_profile(scenario.key, profile)
        loading(screen, "Preparando el entrenamiento...")
        cls = FootballTrainer if scenario.key == 'futbol' else Trainer
        trainer = cls(screen, scenario, cfg, profile, load_brain=(action == 'continue'))
        if trainer.run() == 'quit':
            return True


def show_loop(screen):
    """Campeones -> modo Expo o jugar contra la IA. Devuelve True si hay que salir del programa."""
    from ai.show import ExpoScreen, RaceScreen
    while True:
        action, entry, other_map = gallery(screen)
        if action == 'quit':
            return True
        if action == 'back':
            return False
        screen, scenario = scenario_for_entry(screen, entry, other_map=other_map)
        if scenario is None:
            continue
        cls = RaceScreen if action == 'competir' else ExpoScreen
        if cls(screen, scenario, entry).run() == 'quit':
            return True


def main():
    screen = ui.open_window(CAPTION)
    maze_opts = {'size': 'chico', 'new_every': 0, 'braid': False}
    curriculum_opts = {'rule': 'dominar', 'every': 20, 'threshold': 50, 'tracks': []}
    football_opts = {'team_size': 1, 'opponent': 'none', 'match_steps': 1200, 'ball_control': 'pegada',
                     'random_starts': 0.5}

    while True:
        choice = main_menu(screen)
        if choice == 'quit':
            break
        if choice == 'campeones':
            if show_loop(screen):
                break
            continue
        if choice == 'pista':
            screen, scenario = setup_pista(screen)
        elif choice == 'varias':
            screen, scenario = setup_curriculum(screen, curriculum_opts)
        elif choice == 'futbol':
            screen, scenario = setup_football(screen, football_opts)
        else:
            scenario = maze_settings(screen, maze_opts)
        if scenario == 'quit':
            break
        if scenario is None:
            continue
        if train_loop(screen, scenario):
            break
    pygame.quit()


if __name__ == "__main__":
    main()
