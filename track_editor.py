"""
Track Editor - Sistema de edición de pistas para Self Driving Car AI
- Pintar áreas de pista (blanco) y colisión (negro)
- Ajustar tamaño y forma del pincel (círculo/cuadrado)
- Establecer punto de spawn y dirección de los agentes
- Guardar y cargar pistas personalizadas
- Probar pistas con un agente de prueba
"""

import pygame
import math
import json
import os
from datetime import datetime
import numpy as np

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (60, 60, 60)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 255)


class TrackEditor:
    """Editor de pistas con interfaz gráfica completa"""
    
    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        
        # Área de dibujo (deja espacio para el menú lateral)
        self.canvas_width = width - 300
        self.canvas_height = height
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Track Editor - Self Driving Car AI")
        self.clock = pygame.time.Clock()
        
        # Surface de la pista (canvas de dibujo)
        self.track_surface = pygame.Surface((self.canvas_width, self.canvas_height))
        self.track_surface.fill(BLACK)  # Empezar con todo negro (colisión)
        
        # Estado del editor
        self.paint_mode = 'track'  # 'track' o 'collision'
        self.brush_shape = 'circle'  # 'circle' o 'square'
        self.brush_size = 30
        self.is_painting = False
        
        # Spawn point
        self.spawn_point = None  # (x, y, angle)
        self.setting_spawn = False
        self.spawn_direction_angle = -math.pi / 2  # Hacia arriba por defecto
        
        # UI
        self.menu_x = self.canvas_width
        self.menu_width = 300
        self.buttons = []
        self.sliders = []
        self.create_ui()
        
        # Fuentes
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 32)
        
        # Control
        self.running = True
        self.show_grid = False
        
        # Historial para deshacer (simple)
        self.history = []
        self.max_history = 20
        
        print("\n" + "="*60)
        print("TRACK EDITOR INICIADO")
        print("="*60)
        print("Controles:")
        print("  - Click izquierdo: Pintar")
        print("  - Click derecho: Borrar (pintar colisión)")
        print("  - S: Establecer punto de spawn")
        print("  - G: Mostrar/ocultar grid")
        print("  - Ctrl+Z: Deshacer")
        print("  - ESC: Salir")
        print("="*60 + "\n")
    
    def create_ui(self):
        """Crea los elementos de la interfaz de usuario"""
        menu_x = self.menu_x
        y_offset = 20
        
        # Título
        y_offset += 40
        
        # Sección: Modo de pintura
        y_offset += 20
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 120, 40),
            'label': 'Pista',
            'action': 'set_mode_track',
            'color': WHITE,
            'active': True
        })
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 160, y_offset, 120, 40),
            'label': 'Colisión',
            'action': 'set_mode_collision',
            'color': BLACK,
            'active': False
        })
        y_offset += 60
        
        # Sección: Forma del pincel
        y_offset += 20
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 120, 40),
            'label': 'Círculo',
            'action': 'set_shape_circle',
            'color': BLUE,
            'active': True
        })
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 160, y_offset, 120, 40),
            'label': 'Cuadrado',
            'action': 'set_shape_square',
            'color': BLUE,
            'active': False
        })
        y_offset += 60
        
        # Slider: Tamaño del pincel
        y_offset += 20
        self.sliders.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 260, 20),
            'label': 'Tamaño Pincel',
            'value': 30,
            'min': 5,
            'max': 100,
            'var': 'brush_size'
        })
        y_offset += 60
        
        # Botones de acción
        y_offset += 20
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 260, 40),
            'label': 'Establecer Spawn (S)',
            'action': 'set_spawn',
            'color': GREEN,
            'active': False
        })
        y_offset += 50
        
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 260, 40),
            'label': 'Limpiar Pista',
            'action': 'clear_track',
            'color': RED,
            'active': False
        })
        y_offset += 50
        
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 260, 40),
            'label': 'Guardar Pista',
            'action': 'save_track',
            'color': CYAN,
            'active': False
        })
        y_offset += 50
        
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 260, 40),
            'label': 'Cargar Pista',
            'action': 'load_track',
            'color': ORANGE,
            'active': False
        })
        y_offset += 50
        
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 260, 40),
            'label': 'Probar Pista',
            'action': 'test_track',
            'color': PURPLE,
            'active': False
        })
        y_offset += 50
        
        self.buttons.append({
            'rect': pygame.Rect(menu_x + 20, y_offset, 260, 40),
            'label': 'Crear Pista Oval',
            'action': 'create_oval',
            'color': YELLOW,
            'active': False
        })
    
    def handle_button_click(self, pos):
        """Maneja clicks en botones"""
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                action = button['action']
                
                # Modo de pintura
                if action == 'set_mode_track':
                    self.paint_mode = 'track'
                    self.update_button_states('set_mode_track')
                elif action == 'set_mode_collision':
                    self.paint_mode = 'collision'
                    self.update_button_states('set_mode_collision')
                
                # Forma del pincel
                elif action == 'set_shape_circle':
                    self.brush_shape = 'circle'
                    self.update_button_states('set_shape_circle')
                elif action == 'set_shape_square':
                    self.brush_shape = 'square'
                    self.update_button_states('set_shape_square')
                
                # Acciones
                elif action == 'set_spawn':
                    self.setting_spawn = not self.setting_spawn
                    button['active'] = self.setting_spawn
                    print("Modo establecer spawn:", "ACTIVADO" if self.setting_spawn else "DESACTIVADO")
                
                elif action == 'clear_track':
                    self.save_to_history()
                    self.track_surface.fill(BLACK)
                    self.spawn_point = None
                    print("Pista limpiada")
                
                elif action == 'save_track':
                    self.save_track()
                
                elif action == 'load_track':
                    self.load_track_menu()
                
                elif action == 'test_track':
                    self.test_track()
                
                elif action == 'create_oval':
                    self.create_oval_track()
                
                return True
        return False
    
    def update_button_states(self, active_action):
        """Actualiza el estado activo de los botones en un grupo"""
        for button in self.buttons:
            if button['action'].startswith('set_mode_'):
                button['active'] = (button['action'] == active_action)
            elif button['action'].startswith('set_shape_'):
                button['active'] = (button['action'] == active_action)
    
    def handle_slider_drag(self, pos):
        """Maneja el arrastre de sliders"""
        for slider in self.sliders:
            if slider['rect'].collidepoint(pos):
                # Calcular valor basado en posición del mouse
                rel_x = pos[0] - slider['rect'].x
                ratio = max(0, min(1, rel_x / slider['rect'].width))
                value = slider['min'] + ratio * (slider['max'] - slider['min'])
                slider['value'] = int(value)
                
                # Actualizar variable correspondiente
                if slider['var'] == 'brush_size':
                    self.brush_size = slider['value']
                
                return True
        return False
    
    def paint_at(self, pos):
        """Pinta en la posición especificada"""
        if pos[0] >= self.canvas_width:
            return
        
        color = WHITE if self.paint_mode == 'track' else BLACK
        
        if self.brush_shape == 'circle':
            pygame.draw.circle(self.track_surface, color, pos, self.brush_size)
        else:  # square
            half_size = self.brush_size
            rect = pygame.Rect(pos[0] - half_size, pos[1] - half_size, 
                             half_size * 2, half_size * 2)
            pygame.draw.rect(self.track_surface, color, rect)
    
    def set_spawn_point(self, pos):
        """Establece el punto de spawn"""
        if pos[0] >= self.canvas_width:
            return
        
        self.spawn_point = (pos[0], pos[1], self.spawn_direction_angle)
        self.setting_spawn = False
        
        # Desactivar botón de spawn
        for button in self.buttons:
            if button['action'] == 'set_spawn':
                button['active'] = False
        
        print(f"Spawn point establecido en ({pos[0]}, {pos[1]}) con ángulo {math.degrees(self.spawn_direction_angle):.1f}°")
    
    def save_to_history(self):
        """Guarda el estado actual en el historial"""
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(self.track_surface.copy())
    
    def undo(self):
        """Deshace la última acción"""
        if self.history:
            self.track_surface = self.history.pop()
            print("Deshacer aplicado")
    
    def create_oval_track(self):
        """Crea una pista oval básica como plantilla"""
        self.save_to_history()
        self.track_surface.fill(BLACK)
        
        # Márgenes proporcionales
        margin_x = int(self.canvas_width * 0.1)
        margin_y = int(self.canvas_height * 0.1)
        
        # Óvalo exterior (pista blanca)
        exterior_rect = pygame.Rect(margin_x, margin_y, 
                                   int(self.canvas_width * 0.8), 
                                   int(self.canvas_height * 0.7))
        pygame.draw.ellipse(self.track_surface, WHITE, exterior_rect)
        
        # Óvalo interior (isla negra)
        inner_margin_x = int(self.canvas_width * 0.2)
        inner_margin_y = int(self.canvas_height * 0.25)
        interior_rect = pygame.Rect(inner_margin_x, inner_margin_y,
                                   int(self.canvas_width * 0.6),
                                   int(self.canvas_height * 0.5))
        pygame.draw.ellipse(self.track_surface, BLACK, interior_rect)
        
        # Establecer spawn point por defecto
        self.spawn_point = (self.canvas_width // 2, 
                          int(self.canvas_height * 0.77), 
                          -math.pi / 2)
        
        print("Pista oval creada")
    
    def save_track(self):
        """Guarda la pista actual"""
        if not os.path.exists('tracks'):
            os.makedirs('tracks')
        
        # Generar nombre único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        track_name = f"track_{timestamp}"
        
        # Guardar imagen de la pista
        track_path = f"tracks/{track_name}.png"
        pygame.image.save(self.track_surface, track_path)
        
        # Guardar metadata (spawn point, dimensiones, etc.)
        metadata = {
            'name': track_name,
            'width': self.canvas_width,
            'height': self.canvas_height,
            'spawn_point': self.spawn_point,
            'created': timestamp
        }
        
        metadata_path = f"tracks/{track_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Guardar thumbnail
        thumbnail_size = (200, 150)
        thumbnail = pygame.transform.scale(self.track_surface, thumbnail_size)
        thumbnail_path = f"tracks/{track_name}_thumb.png"
        pygame.image.save(thumbnail, thumbnail_path)
        
        print(f"✓ Pista guardada: {track_name}")
        print(f"  - Imagen: {track_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - Thumbnail: {thumbnail_path}")
    
    def load_track_menu(self):
        """Muestra menú para seleccionar y cargar una pista"""
        if not os.path.exists('tracks'):
            print("No hay pistas guardadas")
            return
        
        # Listar archivos JSON (metadata)
        track_files = [f for f in os.listdir('tracks') if f.endswith('.json')]
        
        if not track_files:
            print("No hay pistas guardadas")
            return
        
        print("\n" + "="*60)
        print("PISTAS DISPONIBLES:")
        print("="*60)
        
        tracks = []
        for i, json_file in enumerate(track_files):
            with open(f"tracks/{json_file}", 'r') as f:
                metadata = json.load(f)
                tracks.append(metadata)
                print(f"{i + 1}. {metadata['name']} - {metadata.get('created', 'N/A')}")
        
        print("="*60)
        print("Cargando la primera pista disponible...")
        
        # Cargar la primera pista como ejemplo
        if tracks:
            self.load_track(tracks[0]['name'])
    
    def load_track(self, track_name):
        """Carga una pista específica"""
        track_path = f"tracks/{track_name}.png"
        metadata_path = f"tracks/{track_name}.json"
        
        if not os.path.exists(track_path) or not os.path.exists(metadata_path):
            print(f"Error: No se encontró la pista {track_name}")
            return
        
        # Cargar imagen
        loaded_surface = pygame.image.load(track_path)
        
        # Cargar metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Aplicar a la superficie actual
        self.track_surface = loaded_surface
        self.spawn_point = tuple(metadata['spawn_point']) if metadata['spawn_point'] else None
        
        print(f"✓ Pista cargada: {track_name}")
        if self.spawn_point:
            print(f"  - Spawn point: ({self.spawn_point[0]}, {self.spawn_point[1]}) @ {math.degrees(self.spawn_point[2]):.1f}°")
    
    def test_track(self):
        """Prueba la pista con un agente simple"""
        if self.spawn_point is None:
            print("Error: Debe establecer un punto de spawn primero")
            return
        
        print("\n" + "="*60)
        print("MODO PRUEBA DE PISTA")
        print("="*60)
        print("Controles:")
        print("  - Flechas: Controlar el auto")
        print("  - ESC: Volver al editor")
        print("="*60 + "\n")
        
        # Importar Car y crear instancia de prueba
        from car import Car
        from environment import Environment
        
        # Crear entorno temporal con la pista actual
        test_env = Environment(width=self.canvas_width, height=self.canvas_height)
        test_env.track = self.track_surface.copy()
        
        # Crear auto
        test_car = Car(test_env)
        test_car.x = self.spawn_point[0]
        test_car.y = self.spawn_point[1]
        test_car.angle = self.spawn_point[2]
        test_car.speed = 2.0
        
        # Loop de prueba
        testing = True
        test_clock = pygame.time.Clock()
        
        while testing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    testing = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        testing = False
            
            # Control manual
            keys = pygame.key.get_pressed()
            action = [0, 0]
            if keys[pygame.K_LEFT]:
                action[0] = -1
            if keys[pygame.K_RIGHT]:
                action[0] = 1
            if keys[pygame.K_UP]:
                action[1] = 1
            
            # Actualizar auto
            test_car.apply_action(action)
            test_car.update()
            
            # Verificar colisión
            px, py = int(test_car.x), int(test_car.y)
            collision = False
            if (px < 0 or px >= self.canvas_width or 
                py < 0 or py >= self.canvas_height):
                collision = True
            elif test_env.track.get_at((px, py)) == (0, 0, 0):
                collision = True
            
            if collision:
                # Reiniciar
                test_car.x = self.spawn_point[0]
                test_car.y = self.spawn_point[1]
                test_car.angle = self.spawn_point[2]
                test_car.speed = 2.0
            
            # Renderizar
            self.screen.blit(test_env.track, (0, 0))
            test_car.draw(self.screen)
            test_car.draw_sensors(self.screen)
            
            # Info
            info_text = self.font.render(f"Velocidad: {test_car.speed:.1f} | ESC para salir", 
                                        True, WHITE)
            pygame.draw.rect(self.screen, BLACK, (5, 5, 400, 35))
            self.screen.blit(info_text, (10, 10))
            
            pygame.display.flip()
            test_clock.tick(60)
        
        print("Modo prueba finalizado")
    
    def draw_ui(self):
        """Dibuja la interfaz de usuario"""
        # Fondo del menú
        menu_rect = pygame.Rect(self.menu_x, 0, self.menu_width, self.height)
        pygame.draw.rect(self.screen, DARK_GRAY, menu_rect)
        pygame.draw.line(self.screen, WHITE, (self.menu_x, 0), (self.menu_x, self.height), 2)
        
        # Título
        title = self.title_font.render("TRACK EDITOR", True, YELLOW)
        self.screen.blit(title, (self.menu_x + 50, 20))
        
        # Secciones
        y_offset = 80
        
        # Modo de pintura
        section = self.font.render("Modo de Pintura:", True, WHITE)
        self.screen.blit(section, (self.menu_x + 20, y_offset))
        y_offset += 40
        
        # Botones de modo
        for button in self.buttons[:2]:
            self.draw_button(button)
        
        y_offset += 80
        
        # Forma del pincel
        section = self.font.render("Forma del Pincel:", True, WHITE)
        self.screen.blit(section, (self.menu_x + 20, y_offset))
        y_offset += 40
        
        # Botones de forma
        for button in self.buttons[2:4]:
            self.draw_button(button)
        
        y_offset += 80
        
        # Slider de tamaño
        section = self.font.render("Tamaño del Pincel:", True, WHITE)
        self.screen.blit(section, (self.menu_x + 20, y_offset))
        y_offset += 30
        
        for slider in self.sliders:
            self.draw_slider(slider)
        
        y_offset += 80
        
        # Acciones
        section = self.font.render("Acciones:", True, WHITE)
        self.screen.blit(section, (self.menu_x + 20, y_offset))
        y_offset += 30
        
        # Resto de botones
        for button in self.buttons[4:]:
            self.draw_button(button)
        
        # Info del estado actual
        y_offset = self.height - 120
        pygame.draw.line(self.screen, WHITE, 
                        (self.menu_x + 10, y_offset), 
                        (self.menu_x + self.menu_width - 10, y_offset), 1)
        y_offset += 10
        
        info_lines = [
            f"Modo: {self.paint_mode.upper()}",
            f"Pincel: {self.brush_shape.upper()} ({self.brush_size}px)",
            f"Spawn: {'SÍ' if self.spawn_point else 'NO'}"
        ]
        
        for line in info_lines:
            text = self.small_font.render(line, True, LIGHT_GRAY)
            self.screen.blit(text, (self.menu_x + 20, y_offset))
            y_offset += 25
    
    def draw_button(self, button):
        """Dibuja un botón"""
        # Color del botón
        if button['active']:
            color = button['color']
            text_color = BLACK if color == WHITE else WHITE
            border_color = YELLOW
        else:
            color = GRAY
            text_color = WHITE
            border_color = WHITE
        
        # Fondo
        pygame.draw.rect(self.screen, color, button['rect'])
        pygame.draw.rect(self.screen, border_color, button['rect'], 2)
        
        # Texto
        text = self.small_font.render(button['label'], True, text_color)
        text_rect = text.get_rect(center=button['rect'].center)
        self.screen.blit(text, text_rect)
    
    def draw_slider(self, slider):
        """Dibuja un slider"""
        # Barra de fondo
        pygame.draw.rect(self.screen, GRAY, slider['rect'])
        pygame.draw.rect(self.screen, WHITE, slider['rect'], 2)
        
        # Indicador de valor
        ratio = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
        indicator_x = slider['rect'].x + int(ratio * slider['rect'].width)
        indicator_rect = pygame.Rect(indicator_x - 5, slider['rect'].y - 5, 10, 30)
        pygame.draw.rect(self.screen, YELLOW, indicator_rect)
        pygame.draw.rect(self.screen, WHITE, indicator_rect, 2)
        
        # Valor actual
        value_text = self.small_font.render(str(slider['value']), True, WHITE)
        self.screen.blit(value_text, (slider['rect'].right + 10, slider['rect'].y))
    
    def draw_canvas(self):
        """Dibuja el canvas de la pista"""
        # Dibujar la pista
        self.screen.blit(self.track_surface, (0, 0))
        
        # Dibujar grid si está activado
        if self.show_grid:
            grid_spacing = 50
            for x in range(0, self.canvas_width, grid_spacing):
                pygame.draw.line(self.screen, DARK_GRAY, (x, 0), (x, self.canvas_height), 1)
            for y in range(0, self.canvas_height, grid_spacing):
                pygame.draw.line(self.screen, DARK_GRAY, (0, y), (self.canvas_width, y), 1)
        
        # Dibujar preview del pincel en la posición del mouse
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[0] < self.canvas_width:
            color = WHITE if self.paint_mode == 'track' else RED
            if self.brush_shape == 'circle':
                pygame.draw.circle(self.screen, color, mouse_pos, self.brush_size, 2)
            else:
                half_size = self.brush_size
                rect = pygame.Rect(mouse_pos[0] - half_size, mouse_pos[1] - half_size,
                                 half_size * 2, half_size * 2)
                pygame.draw.rect(self.screen, color, rect, 2)
        
        # Dibujar spawn point si existe
        if self.spawn_point:
            x, y, angle = self.spawn_point
            
            # Círculo del spawn point
            pygame.draw.circle(self.screen, GREEN, (int(x), int(y)), 15, 3)
            
            # Flecha de dirección
            arrow_length = 40
            end_x = x + arrow_length * math.cos(angle)
            end_y = y + arrow_length * math.sin(angle)
            pygame.draw.line(self.screen, GREEN, (int(x), int(y)), 
                           (int(end_x), int(end_y)), 3)
            
            # Punta de flecha
            arrow_angle1 = angle + math.pi * 0.75
            arrow_angle2 = angle - math.pi * 0.75
            arrow_size = 15
            
            point1_x = end_x + arrow_size * math.cos(arrow_angle1)
            point1_y = end_y + arrow_size * math.sin(arrow_angle1)
            point2_x = end_x + arrow_size * math.cos(arrow_angle2)
            point2_y = end_y + arrow_size * math.sin(arrow_angle2)
            
            pygame.draw.line(self.screen, GREEN, (int(end_x), int(end_y)),
                           (int(point1_x), int(point1_y)), 3)
            pygame.draw.line(self.screen, GREEN, (int(end_x), int(end_y)),
                           (int(point2_x), int(point2_y)), 3)
            
            # Etiqueta
            label = self.small_font.render("SPAWN", True, GREEN)
            self.screen.blit(label, (int(x) - 25, int(y) - 35))
        
        # Instrucciones si está en modo spawn
        if self.setting_spawn:
            instruction = self.font.render("Click para establecer spawn | Flechas para rotar", 
                                         True, YELLOW)
            pygame.draw.rect(self.screen, BLACK, (10, 10, 600, 35))
            self.screen.blit(instruction, (15, 15))
    
    def run(self):
        """Loop principal del editor"""
        dragging_slider = False
        
        while self.running:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_s:
                        self.setting_spawn = not self.setting_spawn
                        print("Modo establecer spawn:", "ACTIVADO" if self.setting_spawn else "DESACTIVADO")
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.undo()
                    
                    # Rotar dirección del spawn
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
                        # Verificar clicks en UI
                        if not self.handle_button_click(event.pos):
                            if self.handle_slider_drag(event.pos):
                                dragging_slider = True
                            elif self.setting_spawn:
                                self.set_spawn_point(event.pos)
                            else:
                                self.save_to_history()
                                self.is_painting = True
                                self.paint_at(event.pos)
                    
                    elif event.button == 3:  # Click derecho (borrar)
                        if event.pos[0] < self.canvas_width:
                            self.save_to_history()
                            self.is_painting = True
                            old_mode = self.paint_mode
                            self.paint_mode = 'collision'
                            self.paint_at(event.pos)
                            self.paint_mode = old_mode
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.is_painting = False
                        dragging_slider = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if dragging_slider:
                        self.handle_slider_drag(event.pos)
                    elif self.is_painting:
                        self.paint_at(event.pos)
            
            # Renderizar
            self.draw_canvas()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        print("\n✓ Track Editor cerrado")


if __name__ == "__main__":
    editor = TrackEditor(width=1920, height=1080)
    editor.run()
