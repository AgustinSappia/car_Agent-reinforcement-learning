"""
Galería de campeones: cerebros entrenados guardados para mostrarlos después
(modo Expo, competir contra la IA) sin tener que entrenar en vivo.

Hay dos tipos de entradas:
- Campeones guardados a mano (botón "A la galería" en el entrenamiento): agentes/galeria/
- El último cerebro de cada perfil y escenario (se guarda solo al entrenar): agentes/<perfil>_<escenario>.npz
"""

import os
import shutil
from datetime import datetime

from ai.brain import read_brain, write_brain
from ai.config import AgentConfig, PROFILES_DIR

GALLERY_DIR = os.path.join(PROFILES_DIR, 'galeria')
KIND_LABELS = {'pista': "Pista", 'laberinto': "Laberinto", 'futbol': "Fútbol"}


def save_champion(path, novice_path=None, name=None):
    """Copia un cerebro a la galería. Devuelve el nombre que se le puso."""
    data = read_brain(path)
    if data is None:
        return None
    os.makedirs(GALLERY_DIR, exist_ok=True)
    meta = dict(data['meta'])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta['name'] = name or f"{KIND_LABELS.get(meta.get('kind'), '')} · {meta.get('map', '')} · gen {meta.get('generation', '?')}"
    meta['saved'] = stamp
    base = os.path.join(GALLERY_DIR, f"campeon_{stamp}")
    write_brain(base + ".npz", data['sizes'], data['W'], data['b'], meta)
    if novice_path and os.path.exists(novice_path):
        shutil.copyfile(novice_path, base + "_gen1.npz")
    return meta['name']


def _entry(path, gallery):
    data = read_brain(path)
    if data is None or not data['meta'].get('kind'):
        return None
    meta = data['meta']
    novice = path[:-4] + "_gen1.npz"
    try:
        cfg = AgentConfig.from_dict(meta.get('config', {}))
    except Exception:
        return None
    if cfg.layer_sizes(meta['kind']) != data['sizes']:
        return None
    if gallery:
        name = meta.get('name', os.path.basename(path))
    else:
        name = f"Perfil {meta.get('profile', '?')} · {KIND_LABELS.get(meta['kind'], meta['kind'])}"
    return {
        'path': path, 'gallery': gallery, 'name': name, 'kind': meta['kind'], 'meta': meta,
        'cfg': cfg, 'sizes': data['sizes'], 'brain': data,
        'novice': read_brain(novice) if os.path.exists(novice) else None,
        'mtime': os.path.getmtime(path),
    }


def list_entries():
    """Campeones de la galería primero (más nuevos arriba), después los últimos de cada perfil"""
    entries = []
    if os.path.isdir(GALLERY_DIR):
        for f in os.listdir(GALLERY_DIR):
            if f.endswith('.npz') and not f.endswith('_gen1.npz'):
                e = _entry(os.path.join(GALLERY_DIR, f), True)
                if e:
                    entries.append(e)
    profile_entries = []
    if os.path.isdir(PROFILES_DIR):
        for f in os.listdir(PROFILES_DIR):
            if f.endswith('.npz') and not f.endswith('_gen1.npz'):
                e = _entry(os.path.join(PROFILES_DIR, f), False)
                if e:
                    profile_entries.append(e)
    entries.sort(key=lambda e: -e['mtime'])
    profile_entries.sort(key=lambda e: -e['mtime'])
    return entries + profile_entries


def delete_entry(entry):
    """Solo se borran campeones de la galería"""
    if not entry['gallery']:
        return False
    for p in (entry['path'], entry['path'][:-4] + "_gen1.npz"):
        if os.path.exists(p):
            os.remove(p)
    return True


def promote(entry):
    """Pasa el último cerebro de un perfil a la galería"""
    return save_champion(entry['path'], entry['path'][:-4] + "_gen1.npz")
