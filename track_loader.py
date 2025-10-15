"""
Track Loader - versión final
Corrige compatibilidad con select_track() y manejo flexible de nombres/rutas.
"""

import pygame
import json
import os


def load_track_data(track_name):
    """
    Carga todos los datos de una pista creada con el editor.
    Acepta tanto nombres simples ("track1") como rutas completas ("tracks/track1.json").
    """
    try:
        base_dir = "tracks"

        # --- Normalizar nombre ---
        track_name = str(track_name).replace("\\", "/")
        track_base = os.path.basename(track_name)
        if track_base.endswith(".json"):
            track_base = track_base[:-5]  # remover .json

        # --- Rutas esperadas ---
        track_path = os.path.join(base_dir, f"{track_base}_track.png")
        metadata_path = os.path.join(base_dir, f"{track_base}.json")

        if not os.path.exists(track_path):
            # Intentar ruta directa si la pista está en otro lugar
            alt_path = f"{track_base}_track.png"
            if os.path.exists(alt_path):
                track_path = alt_path

        if not os.path.exists(metadata_path):
            alt_meta = f"{track_base}.json"
            if os.path.exists(alt_meta):
                metadata_path = alt_meta

        # --- Validar existencia ---
        if not os.path.exists(track_path) or not os.path.exists(metadata_path):
            print(f"✗ No se encontraron archivos para '{track_base}'")
            print(f"  Buscado: {track_path} y {metadata_path}")
            return None

        # --- Cargar metadata ---
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # --- Cargar capa principal ---
        # NO usar convert_alpha() aquí porque pygame.display puede no estar inicializado
        # El Environment se encargará de convertir las surfaces
        track_layer = pygame.image.load(track_path)
        track_width, track_height = track_layer.get_size()

        # --- Función auxiliar para capas opcionales ---
        def load_optional_layer(suffix):
            path = os.path.join(base_dir, f"{track_base}_{suffix}.png")
            if os.path.exists(path):
                try:
                    # NO usar convert_alpha() aquí - se hará en Environment
                    layer = pygame.image.load(path)
                    if layer.get_size() != (track_width, track_height):
                        layer = pygame.transform.smoothscale(layer, (track_width, track_height))
                    return layer
                except Exception as e:
                    print(f"⚠️ Error cargando capa {suffix}: {e}")
                    return None
            return None

        # --- Cargar capas ---
        finish_line_layer = load_optional_layer("finish")
        checkpoint_layer = load_optional_layer("checkpoint")
        speed_zone_layer = load_optional_layer("speed")
        slow_zone_layer = load_optional_layer("slow")

        # --- Armar diccionario final ---
        track_data = {
            "name": track_base,
            "track_layer": track_layer,
            "finish_line_layer": finish_line_layer,
            "checkpoint_layer": checkpoint_layer,
            "speed_zones": speed_zone_layer,
            "slow_zones": slow_zone_layer,
            "spawn_point": tuple(metadata.get("spawn_point", [])) if metadata.get("spawn_point") else None,
            "finish_line": tuple(metadata.get("finish_line", [])) if metadata.get("finish_line") else None,
            "checkpoints": [tuple(cp) for cp in metadata.get("checkpoints", [])],
            "required_laps": int(metadata.get("required_laps", 1)),
            "width": track_width,
            "height": track_height,
            "metadata": metadata,
        }

        print(f"✓ Pista cargada: {track_base}")
        print(f"  - Dimensiones: {track_width}x{track_height}")
        print(f"  - Vueltas requeridas: {track_data['required_laps']}")
        print(f"  - Checkpoints: {len(track_data['checkpoints'])}")
        print(f"  - Spawn: {'Sí' if track_data['spawn_point'] else 'No'}")
        print(f"  - Meta: {'Sí' if track_data['finish_line'] else 'No'}")
        print(f"  - Zonas rápidas: {'Sí' if speed_zone_layer else 'No'}")
        print(f"  - Zonas lentas: {'Sí' if slow_zone_layer else 'No'}")

        return track_data

    except Exception as e:
        print(f"✗ Error cargando pista '{track_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


def list_available_tracks():
    """Devuelve lista de todas las pistas (.json) disponibles en /tracks."""
    if not os.path.exists("tracks"):
        return []
    return [f.replace(".json", "") for f in os.listdir("tracks") if f.endswith(".json")]


if __name__ == "__main__":
    pygame.init()
    tracks = list_available_tracks()
    print(f"Pistas disponibles: {len(tracks)}")
    for t in tracks:
        print(f"  - {t}")
    if tracks:
        print(f"\nCargando: {tracks[0]}")
        d = load_track_data(tracks[0])
        if d:
            print("✓ Carga exitosa")
        else:
            print("✗ Error al cargar")
