"""
Entrenamiento de autos con algoritmo genético.

Flujo:  menú principal → pista (selector) o laberinto (opciones)
        → taller del agente → entrenamiento → vuelve al taller

Uso:  python train.py
"""

import os

import pygame

from ai import ui
from ai.config import AgentConfig, PROFILES_DIR
from ai.menus import main_menu, maze_settings, workshop, loading
from ai.scenarios import PistaScenario
from ai.trainer import Trainer

CAPTION = "Autos que aprenden - Entrenamiento genético"
LAST_PROFILE = os.path.join(PROFILES_DIR, '_ultimo.txt')


def last_profile():
    try:
        with open(LAST_PROFILE) as f:
            name = f.read().strip()
        return name, AgentConfig.load(name)
    except Exception:
        return 'perfil', AgentConfig()


def remember_profile(name):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(LAST_PROFILE, 'w') as f:
        f.write(name)


def choose_track():
    """Abre el selector de pistas (su propia ventana) y devuelve el escenario o None"""
    from track_selector import select_track
    from track_loader import load_track_data

    pygame.quit()
    selected = select_track()
    screen = ui.open_window(CAPTION)
    if not selected:
        return screen, None
    loading(screen, "Preparando la pista...")
    data = load_track_data(selected['name'])
    if not data:
        return screen, None
    data['name'] = selected.get('title') or selected['name']
    return screen, PistaScenario(data)


def main():
    screen = ui.open_window(CAPTION)
    profile, cfg = last_profile()
    maze_opts = {'size': 'chico', 'new_every': 0, 'braid': False}

    while True:
        choice = main_menu(screen)
        if choice == 'quit':
            break
        if choice == 'pista':
            screen, scenario = choose_track()
        else:
            scenario = maze_settings(screen, maze_opts)
        if scenario == 'quit':
            break
        if scenario is None:
            continue

        # Taller ↔ entrenamiento, hasta volver al menú
        while True:
            action, cfg, profile = workshop(screen, scenario, cfg, profile)
            if action == 'quit':
                pygame.quit()
                return
            if action == 'back':
                break
            remember_profile(profile)
            loading(screen, "Preparando el entrenamiento...")
            trainer = Trainer(screen, scenario, cfg, profile, load_brain=(action == 'continue'))
            if trainer.run() == 'quit':
                pygame.quit()
                return
    pygame.quit()


if __name__ == "__main__":
    main()
