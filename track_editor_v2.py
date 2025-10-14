"""
Track Editor V2 - Sistema avanzado de edición de pistas
- Múltiples capas: Pista, Colisión, Meta, Checkpoints, Zonas de velocidad
- Sistema de vueltas con línea de meta
- Zonas especiales (speed zones, slow zones)
- Checkpoints intermedios
"""

import pygame
import math
import json
import os
from datetime import datetime
import numpy as np

# Colores base
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (60, 60, 60)

# Colores para capas especiales
FINISH_LINE_COLOR = (255, 0, 0)      # Rojo para meta
CHECKPOINT_COLOR = (0, 100, 255)      # Azul para checkpoints
SPEED_ZONE_COLOR = (0, 255, 0)        # Verde para zonas rápidas
SLOW_ZONE_COLOR = (255, 255, 0)       # Amarillo para zonas lentas
SPAWN_COLOR = (0, 255, 255)           # Cyan para spawn

# Colores UI
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)


class TrackEditorV2:
    """Editor avanzado de pistas con múltiples capas"""
    
    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        
        # Área de dibujo
        self.canvas_width = width - 350  # Más espacio para menú
        self.canvas_height = height
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Track Editor V2 - Advanced Features")
        self.clock = pygame.time.Clock()
        
        # Capas de la pista
        self.track_layer = pygame.Surface((self.canvas_width, self.canvas_height))
        self.track_layer.fill(BLACK)
        
        self.finish_line_layer = pygame.Surface((self.canvas_width, self.canvas_height))
        self.finish_line_layer.fill((0, 0, 0, 0))
        self.finish_line_layer.set_colorkey((0, 0, 0))
        
        self.checkpoint_layer = pygame.Surface((self.canvas_width, self.canvas_height))
        self.checkpoint_layer.fill((0, 0, 0, 0))
        self.checkpoint_layer.set_colorkey((0, 0, 0))
        
        self.speed_zone_layer = pygame.Surface((self.canvas_width, self.canvas_height))
        self.speed_zone_layer.fill((0, 0, 0, 0))
        self.speed_zone_layer.set_colorkey((0, 0, 0))
        
        self.slow_zone_layer = pygame.Surface((self.canvas_width, self.canvas_height))
        self.slow_zone_layer.fill((0, 0, 0, 0))
        self.slow_zone_layer.set_colorkey((0, 0, 0))
        
        # Estado del editor
        self.current_layer = 'track'  # track, finish, checkpoint, speed_zone, slow_zone
        self.brush_shape = 'circle'
        self.brush_size = 30
        self.is_painting = False
        self.is_erasing = False
        self.layer_selection_mode = False
        
        # Spawn point y finish line
        self.spawn_point = None
        self.setting_spawn = False
        self.spawn_direction_angle = -math.pi / 2
        
        self.finish_line = None  # (x1, y1, x2, y2, angle)
        self.setting_finish_line = False
        self.finish_line_start = None
        
        # Checkpoints
        self.checkpoints = []  # Lista de líneas [(x1, y1, x2, y2), ...]
        self.setting_checkpoint = False
        self.checkpoint_start = None
        
        # Configuración de vueltas
        self.required_laps = 3
        
        # UI
        self.menu_x = self.canvas_width
        self.menu_width = 350
        self.buttons = []
        self.sliders = []
        self.create_ui()
        
        # Fuentes
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 28)
        
        # Control
        self.running = True
        self.show_grid = False
        self.show_layers = {
            'track': True,
            'finish': True,
            'checkpoint': True,
            'speed_zone': True,
            'slow_zone': True
        }
        
        # Historial
        self.history = []
        self.max_history = 20
        
        print("\n" + "="*70)
        print("TRACK EDITOR V2 - ADVANCED FEATURES")
        print("="*70)
        print("Capas disponibles:")
        print("  - Track/Collision (Blanco/Negro)")
        print("  - Finish Line (Rojo) - Meta para vueltas")
        print("  - Checkpoints (Azul) - Puntos intermedios")
        print("  - Speed Zones (Verde) - Zonas de aceleración")
        print("  - Slow Zones (Amarillo) - Zonas de desaceleración")
        print("="*70 + "\n")
    
    def create_ui(self):
        """Crea la interfaz de usuario"""
        menu_x = self.menu_x
        y_offset = 10
        
        # Título
        y_offset += 30
        
        # Sección: Selección de capa
        y_offset += 10
        layer_buttons = [
            ('track', 'Pista', WHITE),
            ('finish', 'Meta', FINISH_LINE_COLOR),
            ('checkpoint', 'Checkpoint', CHECKPOINT_COLOR),
            ('speed_zone', 'Speed Zone', SPEED_ZONE_COLOR),
            ('slow_zone', 'Slow Zone', SLOW_ZONE_COLOR)
        ]
        
        for layer_id, label, color in layer_buttons:
            self.buttons.append({
                'rect': pygame.Rect(menu_x + 10, y_offset, 160, 35),
                'label': label,
                'action': f'set_layer_{layer_id}',
                'color': color,
                'active': (layer_id == 'track')
            })
            y_offset += 40
        
        y_offset += 10
        
        # Forma del pincel
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 10, y_offset, 75, 30),
            'label': 'Círculo',
            'action': 'set_shape_circle',
            'color': BLUE,
            'active': True
        })
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 95, y_offset, 75, 30),
            'label': 'Cuadrado',
            'action': 'set_shape_square',
            'color': BLUE,
            'active': False
        })
        y_offset += 40
        
        # Slider tamaño
        self.sliders.append({
            'rect': pygame.Rect(menu_x + 10, y_offset, 160, 15),
            'label': 'Tamaño',
            'value': 30,
            'min': 5,
            'max': 100,
            'var': 'brush_size'
        })
        y_offset += 35
        
        # Slider vueltas requeridas
        self.sliders.append({
            'rect': pygame.Rect(menu_x + 10, y_offset, 160, 15),
            'label': 'Vueltas',
            'value': 3,
            'min': 1,
            'max': 10,
            'var': 'required_laps'
        })
        y_offset += 40
        
        # Botones de acción
        action_buttons = [
            ('toggle_layer_mode', 'Seleccionar Capa', CYAN),
            ('set_spawn', 'Spawn (S)', CYAN),
            ('set_finish', 'Línea Meta (F)', RED),
            ('set_checkpoint', 'Checkpoint (C)', BLUE),
            ('clear_layer', 'Limpiar Capa', ORANGE),
            ('clear_all', 'Limpiar Todo', RED),
            ('create_oval', 'Pista Oval', YELLOW),
            ('save_track', 'Guardar', GREEN),
            ('load_track', 'Cargar', ORANGE),
            ('test_track', 'Probar', PURPLE)
        ]
        
        for action, label, color in action_buttons:
            self.buttons.append({
                'rect': pygame.Rect(menu_x + 10, y_offset, 160, 35),
                'label': label,
                'action': action,
                'color': color,
                'active': False
            })
            y_offset += 40
    
    def handle_button_click(self, pos):
        """Maneja clicks en botones"""
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                action = button['action']
                
                # Selección de capa
                if action.startswith('set_layer_'):
                    layer = action.replace('set_layer_', '')
                    self.current_layer = layer
                    self.update_button_states(action)
                
                # Forma del pincel
                elif action == 'set_shape_circle':
                    self.brush_shape = 'circle'
                    self.update_button_states(action)
                elif action == 'set_shape_square':
                    self.brush_shape = 'square'
                    self.update_button_states(action)
                
                # Acciones especiales
                elif action == 'toggle_layer_mode':
                    self.layer_selection_mode = not self.layer_selection_mode
                    button['active'] = self.layer_selection_mode

                elif action == 'set_spawn':
                    self.setting_spawn = not self.setting_spawn
                    self.setting_finish_line = False
                    self.setting_checkpoint = False
                    button['active'] = self.setting_spawn

                elif action == 'set_finish':
                    self.setting_finish_line = not self.setting_finish_line
                    self.setting_spawn = False
                    self.setting_checkpoint = False
                    self.finish_line_start = None
                    button['active'] = self.setting_finish_line

                elif action == 'set_checkpoint':
                    self.setting_checkpoint = not self.setting_checkpoint
                    self.setting_spawn = False
                    self.setting_finish_line = False
                    self.checkpoint_start = None
                    button['active'] = self.setting_checkpoint
                
                elif action == 'clear_layer':
                    self.clear_current_layer()
                
                elif action == 'clear_all':
                    self.clear_all_layers()
                
                elif action == 'create_oval':
                    self.create_oval_track()
                
                elif action == 'save_track':
                    self.save_track()
                
                elif action == 'load_track':
                    self.load_track_menu()
                
                elif action == 'test_track':
                    self.test_track()
                
                return True
        return False
    
    def update_button_states(self, active_action):
        """Actualiza estados de botones"""
        for button in self.buttons:
            if button['action'].startswith('set_layer_'):
                button['active'] = (button['action'] == active_action)
            elif button['action'].startswith('set_shape_'):
                button['active'] = (button['action'] == active_action)
    
    def handle_slider_drag(self, pos):
        """Maneja arrastre de sliders"""
        for slider in self.sliders:
            if slider['rect'].collidepoint(pos):
                rel_x = pos[0] - slider['rect'].x
                ratio = max(0, min(1, rel_x / slider['rect'].width))
                value = slider['min'] + ratio * (slider['max'] - slider['min'])
                slider['value'] = int(value)
                
                if slider['var'] == 'brush_size':
                    self.brush_size = slider['value']
                elif slider['var'] == 'required_laps':
                    self.required_laps = slider['value']
                
                return True
        return False
    
    def paint_at(self, pos):
        """Pinta en la capa actual"""
        if pos[0] >= self.canvas_width:
            return

        # Seleccionar capa y color
        if self.current_layer == 'track':
            layer = self.track_layer
            color = WHITE
        elif self.current_layer == 'finish':
            layer = self.finish_line_layer
            color = FINISH_LINE_COLOR
        elif self.current_layer == 'checkpoint':
            layer = self.checkpoint_layer
            color = CHECKPOINT_COLOR
        elif self.current_layer == 'speed_zone':
            layer = self.speed_zone_layer
            color = SPEED_ZONE_COLOR
        elif self.current_layer == 'slow_zone':
            layer = self.slow_zone_layer
            color = SLOW_ZONE_COLOR
        else:
            return

        # Pintar
        if self.brush_shape == 'circle':
            pygame.draw.circle(layer, color, pos, self.brush_size)
        else:
            half_size = self.brush_size
            rect = pygame.Rect(pos[0] - half_size, pos[1] - half_size,
                             half_size * 2, half_size * 2)
            pygame.draw.rect(layer, color, rect)

    def erase_at(self, pos):
        """Borra en la capa actual"""
        if pos[0] >= self.canvas_width:
            return

        # Seleccionar capa
        if self.current_layer == 'track':
            layer = self.track_layer
            color = BLACK
        elif self.current_layer == 'finish':
            layer = self.finish_line_layer
            color = (0, 0, 0, 0)
        elif self.current_layer == 'checkpoint':
            layer = self.checkpoint_layer
            color = (0, 0, 0, 0)
        elif self.current_layer == 'speed_zone':
            layer = self.speed_zone_layer
            color = (0, 0, 0, 0)
        elif self.current_layer == 'slow_zone':
            layer = self.slow_zone_layer
            color = (0, 0, 0, 0)
        else:
            return

        # Borrar
        if self.brush_shape == 'circle':
            pygame.draw.circle(layer, color, pos, self.brush_size)
        else:
            half_size = self.brush_size
            rect = pygame.Rect(pos[0] - half_size, pos[1] - half_size,
                             half_size * 2, half_size * 2)
            pygame.draw.rect(layer, color, rect)
    
    def set_spawn_point(self, pos):
        """Establece punto de spawn"""
        if pos[0] >= self.canvas_width:
            return
        
        self.spawn_point = (pos[0], pos[1], self.spawn_direction_angle)
        self.setting_spawn = False
        
        for button in self.buttons:
            if button['action'] == 'set_spawn':
                button['active'] = False
        
        print(f"✓ Spawn establecido en ({pos[0]}, {pos[1]}) @ {math.degrees(self.spawn_direction_angle):.1f}°")
    
    def handle_finish_line_click(self, pos):
        """Maneja clicks para establecer línea de meta"""
        if pos[0] >= self.canvas_width:
            return
        
        if self.finish_line_start is None:
            self.finish_line_start = pos
            print("Punto inicial de meta establecido. Click para punto final.")
        else:
            # Calcular ángulo de la línea
            dx = pos[0] - self.finish_line_start[0]
            dy = pos[1] - self.finish_line_start[1]
            angle = math.atan2(dy, dx)
            
            self.finish_line = (
                self.finish_line_start[0],
                self.finish_line_start[1],
                pos[0],
                pos[1],
                angle
            )
            
            # Dibujar en la capa
            pygame.draw.line(self.finish_line_layer, FINISH_LINE_COLOR,
                           self.finish_line_start, pos, 10)
            
            self.finish_line_start = None
            self.setting_finish_line = False
            
            for button in self.buttons:
                if button['action'] == 'set_finish':
                    button['active'] = False
            
            print(f"✓ Línea de meta establecida")
    
    def handle_checkpoint_click(self, pos):
        """Maneja clicks para establecer checkpoints"""
        if pos[0] >= self.canvas_width:
            return
        
        if self.checkpoint_start is None:
            self.checkpoint_start = pos
            print("Punto inicial de checkpoint. Click para punto final.")
        else:
            checkpoint = (
                self.checkpoint_start[0],
                self.checkpoint_start[1],
                pos[0],
                pos[1]
            )
            self.checkpoints.append(checkpoint)
            
            # Dibujar en la capa
            pygame.draw.line(self.checkpoint_layer, CHECKPOINT_COLOR,
                           self.checkpoint_start, pos, 8)
            
            self.checkpoint_start = None
            print(f"✓ Checkpoint {len(self.checkpoints)} añadido")
    
    def clear_current_layer(self):
        """Limpia la capa actual"""
        if self.current_layer == 'track':
            self.track_layer.fill(BLACK)
        elif self.current_layer == 'finish':
            self.finish_line_layer.fill((0, 0, 0))
            self.finish_line = None
        elif self.current_layer == 'checkpoint':
            self.checkpoint_layer.fill((0, 0, 0))
            self.checkpoints = []
        elif self.current_layer == 'speed_zone':
            self.speed_zone_layer.fill((0, 0, 0))
        elif self.current_layer == 'slow_zone':
            self.slow_zone_layer.fill((0, 0, 0))
        
        print(f"✓ Capa '{self.current_layer}' limpiada")
    
    def clear_all_layers(self):
        """Limpia todas las capas"""
        self.track_layer.fill(BLACK)
        self.finish_line_layer.fill((0, 0, 0))
        self.checkpoint_layer.fill((0, 0, 0))
        self.speed_zone_layer.fill((0, 0, 0))
        self.slow_zone_layer.fill((0, 0, 0))
        
        self.spawn_point = None
        self.finish_line = None
        self.checkpoints = []
        
        print("✓ Todas las capas limpiadas")
    
    def create_oval_track(self):
        """Crea pista oval con meta y checkpoints"""
        self.clear_all_layers()
        
        # Pista oval
        margin_x = int(self.canvas_width * 0.1)
        margin_y = int(self.canvas_height * 0.1)
        
        exterior_rect = pygame.Rect(margin_x, margin_y,
                                   int(self.canvas_width * 0.8),
                                   int(self.canvas_height * 0.7))
        pygame.draw.ellipse(self.track_layer, WHITE, exterior_rect)
        
        inner_margin_x = int(self.canvas_width * 0.2)
        inner_margin_y = int(self.canvas_height * 0.25)
        interior_rect = pygame.Rect(inner_margin_x, inner_margin_y,
                                   int(self.canvas_width * 0.6),
                                   int(self.canvas_height * 0.5))
        pygame.draw.ellipse(self.track_layer, BLACK, interior_rect)
        
        # Spawn point
        self.spawn_point = (self.canvas_width // 2,
                          int(self.canvas_height * 0.77),
                          -math.pi / 2)
        
        # Línea de meta (horizontal en la parte inferior)
        finish_y = int(self.canvas_height * 0.75)
        finish_x1 = int(self.canvas_width * 0.35)
        finish_x2 = int(self.canvas_width * 0.65)
        
        self.finish_line = (finish_x1, finish_y, finish_x2, finish_y, 0)
        pygame.draw.line(self.finish_line_layer, FINISH_LINE_COLOR,
                       (finish_x1, finish_y), (finish_x2, finish_y), 10)
        
        # Checkpoints (2 checkpoints en lados opuestos)
        # Checkpoint 1 (arriba)
        cp1_y = int(self.canvas_height * 0.2)
        cp1_x1 = int(self.canvas_width * 0.35)
        cp1_x2 = int(self.canvas_width * 0.65)
        self.checkpoints.append((cp1_x1, cp1_y, cp1_x2, cp1_y))
        pygame.draw.line(self.checkpoint_layer, CHECKPOINT_COLOR,
                       (cp1_x1, cp1_y), (cp1_x2, cp1_y), 8)
        
        # Checkpoint 2 (medio)
        cp2_y = int(self.canvas_height * 0.475)
        cp2_x1 = int(self.canvas_width * 0.15)
        cp2_x2 = int(self.canvas_width * 0.25)
        self.checkpoints.append((cp2_x1, cp2_y, cp2_x2, cp2_y))
        pygame.draw.line(self.checkpoint_layer, CHECKPOINT_COLOR,
                       (cp2_x1, cp2_y), (cp2_x2, cp2_y), 8)
        
        print("✓ Pista oval creada con meta y checkpoints")
    
    def save_track(self):
        """Guarda la pista con todas las capas"""
        if not os.path.exists('tracks'):
            os.makedirs('tracks')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        track_name = f"track_{timestamp}"
        
        # Combinar todas las capas para la imagen principal
        combined = self.track_layer.copy()
        if self.show_layers['speed_zone']:
            combined.blit(self.speed_zone_layer, (0, 0), special_flags=pygame.BLEND_ADD)
        if self.show_layers['slow_zone']:
            combined.blit(self.slow_zone_layer, (0, 0), special_flags=pygame.BLEND_ADD)
        if self.show_layers['checkpoint']:
            combined.blit(self.checkpoint_layer, (0, 0), special_flags=pygame.BLEND_ADD)
        if self.show_layers['finish']:
            combined.blit(self.finish_line_layer, (0, 0), special_flags=pygame.BLEND_ADD)
        
        # Guardar imagen principal
        track_path = f"tracks/{track_name}.png"
        pygame.image.save(combined, track_path)
        
        # Guardar capas individuales
        pygame.image.save(self.track_layer, f"tracks/{track_name}_track.png")
        pygame.image.save(self.finish_line_layer, f"tracks/{track_name}_finish.png")
        pygame.image.save(self.checkpoint_layer, f"tracks/{track_name}_checkpoint.png")
        pygame.image.save(self.speed_zone_layer, f"tracks/{track_name}_speed.png")
        pygame.image.save(self.slow_zone_layer, f"tracks/{track_name}_slow.png")
        
        # Metadata
        metadata = {
            'name': track_name,
            'width': self.canvas_width,
            'height': self.canvas_height,
            'spawn_point': self.spawn_point,
            'finish_line': self.finish_line,
            'checkpoints': self.checkpoints,
            'required_laps': self.required_laps,
            'created': timestamp,
            'version': 2  # Versión del editor
        }
        
        with open(f"tracks/{track_name}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Thumbnail
        thumbnail = pygame.transform.scale(combined, (200, 150))
        pygame.image.save(thumbnail, f"tracks/{track_name}_thumb.png")
        
        print(f"✓ Pista guardada: {track_name}")
        print(f"  - Vueltas requeridas: {self.required_laps}")
        print(f"  - Checkpoints: {len(self.checkpoints)}")
    
    def load_track_menu(self):
        """Carga una pista (simplificado)"""
        if not os.path.exists('tracks'):
            print("No hay pistas guardadas")
            return
        
        track_files = [f for f in os.listdir('tracks') if f.endswith('.json')]
        if not track_files:
            print("No hay pistas guardadas")
            return
        
        # Cargar la primera pista
        with open(f"tracks/{track_files[0]}", 'r') as f:
            metadata = json.load(f)
        
        self.load_track(metadata['name'])
    
    def load_track(self, track_name):
        """Carga una pista específica"""
        try:
            # Cargar capas
            self.track_layer = pygame.image.load(f"tracks/{track_name}_track.png")
            
            if os.path.exists(f"tracks/{track_name}_finish.png"):
                self.finish_line_layer = pygame.image.load(f"tracks/{track_name}_finish.png")
            if os.path.exists(f"tracks/{track_name}_checkpoint.png"):
                self.checkpoint_layer = pygame.image.load(f"tracks/{track_name}_checkpoint.png")
            if os.path.exists(f"tracks/{track_name}_speed.png"):
                self.speed_zone_layer = pygame.image.load(f"tracks/{track_name}_speed.png")
            if os.path.exists(f"tracks/{track_name}_slow.png"):
                self.slow_zone_layer = pygame.image.load(f"tracks/{track_name}_slow.png")
            
            # Cargar metadata
            with open(f"tracks/{track_name}.json", 'r') as f:
                metadata = json.load(f)
            
            self.spawn_point = tuple(metadata['spawn_point']) if metadata.get('spawn_point') else None
            self.finish_line = tuple(metadata['finish_line']) if metadata.get('finish_line') else None
            self.checkpoints = [tuple(cp) for cp in metadata.get('checkpoints', [])]
            self.required_laps = metadata.get('required_laps', 3)
            
            # Actualizar slider de vueltas
            for slider in self.sliders:
                if slider['var'] == 'required_laps':
                    slider['value'] = self.required_laps
            
            print(f"✓ Pista cargada: {track_name}")
            print(f"  - Vueltas: {self.required_laps}")
            print(f"  - Checkpoints: {len(self.checkpoints)}")
            
        except Exception as e:
            print(f"Error cargando pista: {e}")
    
    def test_track(self):
        """Prueba la pista con control manual del auto"""
        if not self.spawn_point:
            print("Error: Debe establecer un punto de spawn primero")
            return

        print("\n" + "="*60)
        print("MODO PRUEBA - Control Manual del Auto")
        print("="*60)
        print("Controles: ↑ Acelerar, ← Girar Izquierda, → Girar Derecha, ESC para salir")
        print("La pista se probará con el auto en modo manual")
        print("="*60 + "\n")

        # Crear entorno de prueba
        from environment import Environment
        import numpy as np

        # Preparar datos de pista personalizada
        custom_track_data = {
            'track_layer': self.track_layer,
            'finish_line': self.finish_line,
            'checkpoints': self.checkpoints,
            'speed_zones': self.speed_zone_layer if hasattr(self, 'speed_zone_layer') else None,
            'slow_zones': self.slow_zone_layer if hasattr(self, 'slow_zone_layer') else None,
            'required_laps': self.required_laps,
            'spawn_point': self.spawn_point
        }

        # Crear entorno con pista personalizada
        test_env = Environment(width=self.canvas_width, height=self.canvas_height,
                              custom_track_data=custom_track_data)

        # Reset y dar velocidad inicial
        observation = test_env.reset()
        test_env.car.speed = 2.0

        # Variables para simulación
        testing = True
        test_font = pygame.font.Font(None, 24)

        while testing and self.running:
            # Manejar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    testing = False
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        testing = False

            # Leer teclas para control manual
            keys = pygame.key.get_pressed()
            action = [0, 0]  # (steering, throttle)
            if keys[pygame.K_LEFT]:
                action[0] = -1  # Girar izquierda
            if keys[pygame.K_RIGHT]:
                action[0] = 1   # Girar derecha
            if keys[pygame.K_UP]:
                action[1] = 1   # Acelerar

            # Ejecutar un paso en el entorno
            observation, reward, done, info = test_env.step(action)

            # Reiniciar automáticamente al chocar
            if done:
                observation = test_env.reset()
                test_env.car.speed = 2.0

            # Renderizar el entorno de prueba
            test_env.render(self.screen)

            # Mostrar info en pantalla
            if done:
                text = test_font.render("¡Choque! Reiniciando...", True, (255, 0, 0))
                self.screen.blit(text, (10, 10))
            else:
                x, y = info['position']
                min_sensor = info.get('min_sensor', 0)
                text_vel = test_font.render(f"Vel: {info['speed']:.1f} | Pos: ({x:.0f}, {y:.0f})", True, (255, 255, 255))
                text_sensor = test_font.render(f"Sensor mín: {min_sensor:.0f} | Recompensa: {reward:.2f}", True, (255, 255, 255))

                # Fondo negro para mejor legibilidad
                pygame.draw.rect(self.screen, (0, 0, 0), (5, 5, 400, 50))
                self.screen.blit(text_vel, (10, 10))
                self.screen.blit(text_sensor, (10, 30))

            # Actualizar pantalla
            pygame.display.flip()
            self.clock.tick(60)

        print("Prueba finalizada. Regresando al editor...")
    
    def draw(self):
        """Dibuja todo"""
        self.screen.fill(DARK_GRAY)
        
        # Dibujar capas
        self.screen.blit(self.track_layer, (0, 0))
        
        if self.show_layers['speed_zone']:
            self.screen.blit(self.speed_zone_layer, (0, 0))
        if self.show_layers['slow_zone']:
            self.screen.blit(self.slow_zone_layer, (0, 0))
        if self.show_layers['checkpoint']:
            self.screen.blit(self.checkpoint_layer, (0, 0))
        if self.show_layers['finish']:
            self.screen.blit(self.finish_line_layer, (0, 0))
        
        # Grid
        if self.show_grid:
            for x in range(0, self.canvas_width, 50):
                pygame.draw.line(self.screen, DARK_GRAY, (x, 0), (x, self.canvas_height), 1)
            for y in range(0, self.canvas_height, 50):
                pygame.draw.line(self.screen, DARK_GRAY, (0, y), (self.canvas_width, y), 1)
        
        # Preview del pincel
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[0] < self.canvas_width and not (self.setting_spawn or self.setting_finish_line or self.setting_checkpoint):
            color = self.get_current_layer_color()
            if self.brush_shape == 'circle':
                pygame.draw.circle(self.screen, color, mouse_pos, self.brush_size, 2)
            else:
                half = self.brush_size
                pygame.draw.rect(self.screen, color,
                               (mouse_pos[0]-half, mouse_pos[1]-half, half*2, half*2), 2)
        
        # Spawn point
        if self.spawn_point:
            x, y, angle = self.spawn_point
            pygame.draw.circle(self.screen, SPAWN_COLOR, (int(x), int(y)), 15, 3)
            
            end_x = x + 40 * math.cos(angle)
            end_y = y + 40 * math.sin(angle)
            pygame.draw.line(self.screen, SPAWN_COLOR, (int(x), int(y)),
                           (int(end_x), int(end_y)), 3)
        
        # Línea temporal para finish line o checkpoint
        if self.setting_finish_line and self.finish_line_start:
            pygame.draw.line(self.screen, RED, self.finish_line_start, mouse_pos, 10)
        
        if self.setting_checkpoint and self.checkpoint_start:
            pygame.draw.line(self.screen, BLUE, self.checkpoint_start, mouse_pos, 8)
        
        # UI
        self.draw_ui()
    
    def get_current_layer_color(self):
        """Obtiene el color de la capa actual"""
        colors = {
            'track': WHITE,
            'finish': FINISH_LINE_COLOR,
            'checkpoint': CHECKPOINT_COLOR,
            'speed_zone': SPEED_ZONE_COLOR,
            'slow_zone': SLOW_ZONE_COLOR
        }
        return colors.get(self.current_layer, WHITE)
    
    def draw_ui(self):
        """Dibuja la interfaz de usuario"""
        # Fondo del menú
        menu_rect = pygame.Rect(self.menu_x, 0, self.menu_width, self.height)
        pygame.draw.rect(self.screen, DARK_GRAY, menu_rect)
        pygame.draw.line(self.screen, WHITE, (self.menu_x, 0), (self.menu_x, self.height), 2)
        
        # Título
        title = self.title_font.render("TRACK EDITOR V2", True, YELLOW)
        self.screen.blit(title, (self.menu_x + 80, 10))
        
        # Sección capas
        y_offset = 50
        section = self.small_font.render("Capas:", True, WHITE)
        self.screen.blit(section, (self.menu_x + 10, y_offset))
        
        # Dibujar botones
        for button in self.buttons:
            self.draw_button(button)
        
        # Dibujar sliders
        for slider in self.sliders:
            self.draw_slider(slider)
        
        # Info del estado
        y_offset = self.height - 150
        pygame.draw.line(self.screen, WHITE,
                        (self.menu_x + 10, y_offset),
                        (self.menu_x + self.menu_width - 10, y_offset), 1)
        y_offset += 10
        
        info_lines = [
            f"Capa: {self.current_layer.upper()}",
            f"Pincel: {self.brush_shape} ({self.brush_size}px)",
            f"Vueltas: {self.required_laps}",
            f"Checkpoints: {len(self.checkpoints)}",
            f"Spawn: {'SÍ' if self.spawn_point else 'NO'}",
            f"Meta: {'SÍ' if self.finish_line else 'NO'}"
        ]
        
        for line in info_lines:
            text = self.small_font.render(line, True, LIGHT_GRAY)
            self.screen.blit(text, (self.menu_x + 10, y_offset))
            y_offset += 20
    
    def draw_button(self, button):
        """Dibuja un botón"""
        if button['active']:
            color = button['color']
            text_color = BLACK if color in [WHITE, YELLOW, SPEED_ZONE_COLOR] else WHITE
            border_color = YELLOW
        else:
            color = GRAY
            text_color = WHITE
            border_color = WHITE
        
        pygame.draw.rect(self.screen, color, button['rect'])
        pygame.draw.rect(self.screen, border_color, button['rect'], 2)
        
        text = self.small_font.render(button['label'], True, text_color)
        text_rect = text.get_rect(center=button['rect'].center)
        self.screen.blit(text, text_rect)
    
    def draw_slider(self, slider):
        """Dibuja un slider"""
        # Etiqueta
        label = self.small_font.render(f"{slider['label']}: {slider['value']}", True, WHITE)
        self.screen.blit(label, (slider['rect'].x, slider['rect'].y - 18))
        
        # Barra
        pygame.draw.rect(self.screen, GRAY, slider['rect'])
        pygame.draw.rect(self.screen, WHITE, slider['rect'], 2)
        
        # Indicador
        ratio = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
        indicator_x = slider['rect'].x + int(ratio * slider['rect'].width)
        indicator_rect = pygame.Rect(indicator_x - 5, slider['rect'].y - 5, 10, 25)
        pygame.draw.rect(self.screen, YELLOW, indicator_rect)
        pygame.draw.rect(self.screen, WHITE, indicator_rect, 2)
    
    def run(self):
        """Loop principal"""
        dragging_slider = False
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_s:
                        self.setting_spawn = not self.setting_spawn
                    elif event.key == pygame.K_f:
                        self.setting_finish_line = not self.setting_finish_line
                        self.finish_line_start = None
                    elif event.key == pygame.K_c:
                        self.setting_checkpoint = not self.setting_checkpoint
                        self.checkpoint_start = None
                    
                    # Rotar spawn
                    elif self.setting_spawn:
                        if event.key == pygame.K_LEFT:
                            self.spawn_direction_angle -= math.pi / 8
                        elif event.key == pygame.K_RIGHT:
                            self.spawn_direction_angle += math.pi / 8
                        elif event.key == pygame.K_UP:
                            self.spawn_direction_angle = -math.pi / 2
                        elif event.key == pygame.K_DOWN:
                            self.spawn_direction_angle = math.pi / 2
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Click izquierdo
                        if not self.handle_button_click(event.pos):
                            if self.handle_slider_drag(event.pos):
                                dragging_slider = True
                            elif self.setting_spawn:
                                self.set_spawn_point(event.pos)
                            elif self.setting_finish_line:
                                self.handle_finish_line_click(event.pos)
                            elif self.setting_checkpoint:
                                self.handle_checkpoint_click(event.pos)
                            else:
                                self.is_painting = True
                                self.is_erasing = False
                                self.paint_at(event.pos)

                    elif event.button == 3:  # Click derecho (borrar)
                        if event.pos[0] < self.canvas_width:
                            self.is_painting = True
                            self.is_erasing = True
                            self.erase_at(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.is_painting = False
                        dragging_slider = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if dragging_slider:
                        self.handle_slider_drag(event.pos)
                    elif self.is_painting:
                        if self.is_erasing:
                            self.erase_at(event.pos)
                        else:
                            self.paint_at(event.pos)
            
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        print("\n✓ Track Editor V2 cerrado")


if __name__ == "__main__":
    editor = TrackEditorV2(width=1920, height=1080)
    editor.run()
