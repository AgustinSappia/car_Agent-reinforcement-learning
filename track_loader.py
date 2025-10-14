"""
Track Loader - Utilidad para cargar pistas guardadas
Carga todas las capas y metadata de una pista
"""

import pygame
import json
import os


def load_track_data(track_name):
    """
    Carga todos los datos de una pista guardada.
    
    Args:
        track_name: Nombre de la pista (sin extensión)
    
    Returns:
        dict con todas las capas y metadata, o None si hay error
    """
    try:
        # Verificar que existan los archivos necesarios
        track_path = f"tracks/{track_name}_track.png"
        metadata_path = f"tracks/{track_name}.json"
        
        if not os.path.exists(track_path) or not os.path.exists(metadata_path):
            print(f"Error: No se encontraron archivos para {track_name}")
            return None
        
        # Cargar metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Cargar capa principal (track)
        track_layer = pygame.image.load(track_path)
        
        # Cargar capas opcionales
        finish_line_layer = None
        checkpoint_layer = None
        speed_zone_layer = None
        slow_zone_layer = None
        
        if os.path.exists(f"tracks/{track_name}_finish.png"):
            finish_line_layer = pygame.image.load(f"tracks/{track_name}_finish.png")
        
        if os.path.exists(f"tracks/{track_name}_checkpoint.png"):
            checkpoint_layer = pygame.image.load(f"tracks/{track_name}_checkpoint.png")
        
        if os.path.exists(f"tracks/{track_name}_speed.png"):
            speed_zone_layer = pygame.image.load(f"tracks/{track_name}_speed.png")
        
        if os.path.exists(f"tracks/{track_name}_slow.png"):
            slow_zone_layer = pygame.image.load(f"tracks/{track_name}_slow.png")
        
        # Construir diccionario de datos
        track_data = {
            'name': track_name,
            'track_layer': track_layer,
            'finish_line_layer': finish_line_layer,
            'checkpoint_layer': checkpoint_layer,
            'speed_zones': speed_zone_layer,
            'slow_zones': slow_zone_layer,
            'spawn_point': tuple(metadata['spawn_point']) if metadata.get('spawn_point') else None,
            'finish_line': tuple(metadata['finish_line']) if metadata.get('finish_line') else None,
            'checkpoints': [tuple(cp) for cp in metadata.get('checkpoints', [])],
            'required_laps': metadata.get('required_laps', 3),
            'width': metadata.get('width'),
            'height': metadata.get('height'),
            'metadata': metadata
        }
        
        print(f"✓ Pista cargada: {track_name}")
        print(f"  - Dimensiones: {track_data['width']}x{track_data['height']}")
        print(f"  - Vueltas requeridas: {track_data['required_laps']}")
        print(f"  - Checkpoints: {len(track_data['checkpoints'])}")
        print(f"  - Spawn: {'Sí' if track_data['spawn_point'] else 'No'}")
        print(f"  - Meta: {'Sí' if track_data['finish_line'] else 'No'}")
        
        return track_data
        
    except Exception as e:
        print(f"Error cargando pista {track_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def list_available_tracks():
    """
    Lista todas las pistas disponibles en el directorio tracks/
    
    Returns:
        Lista de nombres de pistas (sin extensión)
    """
    if not os.path.exists('tracks'):
        return []
    
    track_files = [f for f in os.listdir('tracks') if f.endswith('.json')]
    track_names = [f.replace('.json', '').replace('track_', 'track_') for f in track_files]
    
    # Extraer solo el nombre base
    track_names = [f.replace('.json', '') for f in track_files]
    track_names = [name.replace('.json', '') for name in track_names]
    
    return track_names


if __name__ == "__main__":
    # Test
    tracks = list_available_tracks()
    print(f"Pistas disponibles: {len(tracks)}")
    for track in tracks:
        print(f"  - {track}")
    
    if tracks:
        print(f"\nCargando primera pista: {tracks[0]}")
        data = load_track_data(tracks[0])
        if data:
            print("✓ Carga exitosa")
