#Track Selector - Menú gráfico para seleccionar pistas guardadas
#- Muestra miniaturas de pistas disponibles
#- Permite seleccionar pista para entrenamiento
#- Opción para eliminar pistas
#- Botón para crear nueva pista


import pygame
import json
import os
import sys
from datetime import datetime

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


class TrackSelector:
    """Selector gráfico de pistas con miniaturas"""
    
    def __init__(self, width=1280, height=720):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Selector de Pistas - Self Driving Car AI")
        self.clock = pygame.time.Clock()
        
        # Fuentes
        self.title_font = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Estado
        self.running = True
        self.selected_track = None
        self.tracks = []
        self.track_buttons = []
        self.scroll_offset = 0
        self.max_scroll = 0
        
        # Botones principales
        self.create_new_button = None
        self.start_training_button = None
        self.delete_mode = False
        
        # Cargar pistas disponibles
        self.load_tracks()
        self.create_ui()
        
        print("\n" + "="*60)
        print("SELECTOR DE PISTAS")
        print("="*60)
        print(f"Pistas disponibles: {len(self.tracks)}")
        print("="*60 + "\n")
    
    def load_tracks(self):
        """Carga todas las pistas disponibles"""
        self.tracks = []
        
        if not os.path.exists('tracks'):
            os.makedirs('tracks')
            return
        
        # Buscar archivos JSON (metadata)
        track_files = [f for f in os.listdir('tracks') if f.endswith('.json')]
        
        for json_file in track_files:
            try:
                with open(f"tracks/{json_file}", 'r') as f:
                    metadata = json.load(f)
                    
                    # Verificar que existan los archivos necesarios
                    track_name = metadata['name']
                    track_path = f"tracks/{track_name}.png"
                    thumb_path = f"tracks/{track_name}_thumb.png"
                    
                    if os.path.exists(track_path):
                        # Cargar thumbnail
                        if os.path.exists(thumb_path):
                            thumbnail = pygame.image.load(thumb_path)
                        else:
                            # Crear thumbnail si no existe
                            track_img = pygame.image.load(track_path)
                            thumbnail = pygame.transform.scale(track_img, (200, 150))
                            pygame.image.save(thumbnail, thumb_path)
                        
                        self.tracks.append({
                            'name': track_name,
                            'metadata': metadata,
                            'thumbnail': thumbnail,
                            'path': track_path
                        })
            except Exception as e:
                print(f"Error cargando pista {json_file}: {e}")
        
        # Ordenar por fecha de creación (más reciente primero)
        self.tracks.sort(key=lambda x: x['metadata'].get('created', ''), reverse=True)
    
    def create_ui(self):
        """Crea los elementos de la interfaz"""
        # Botón para crear nueva pista
        self.create_new_button = {
            'rect': pygame.Rect(50, self.height - 80, 250, 60),
            'label': 'Crear Nueva Pista',
            'color': GREEN
        }
        
        # Botón para iniciar entrenamiento
        self.start_training_button = {
            'rect': pygame.Rect(self.width - 300, self.height - 80, 250, 60),
            'label': 'Iniciar Entrenamiento',
            'color': BLUE
        }
        
        # Botón para modo eliminar
        self.delete_button = {
            'rect': pygame.Rect(self.width // 2 - 125, self.height - 80, 250, 60),
            'label': 'Modo Eliminar',
            'color': RED
        }
        
        # Crear botones para cada pista (grid de 4 columnas)
        self.track_buttons = []
        cols = 4
        thumb_width = 200
        thumb_height = 150
        padding = 30
        start_y = 120
        
        for i, track in enumerate(self.tracks):
            row = i // cols
            col = i % cols
            
            x = 50 + col * (thumb_width + padding)
            y = start_y + row * (thumb_height + padding + 60)
            
            self.track_buttons.append({
                'rect': pygame.Rect(x, y, thumb_width, thumb_height + 60),
                'track': track,
                'thumbnail_rect': pygame.Rect(x, y, thumb_width, thumb_height),
                'delete_rect': pygame.Rect(x + thumb_width - 30, y, 30, 30)
            })
        
        # Calcular scroll máximo
        if self.track_buttons:
            last_button = self.track_buttons[-1]
            self.max_scroll = max(0, last_button['rect'].bottom - (self.height - 120))
    
    def handle_click(self, pos):
        """Maneja clicks del mouse"""
        adjusted_pos = (pos[0], pos[1] + self.scroll_offset)
        
        # Click en botón crear nueva
        if self.create_new_button['rect'].collidepoint(pos):
            self.launch_track_editor()
            return
        
        # Click en botón iniciar entrenamiento
        if self.start_training_button['rect'].collidepoint(pos):
            if self.selected_track:
                self.running = False
            return
        
        # Click en botón modo eliminar
        if self.delete_button['rect'].collidepoint(pos):
            self.delete_mode = not self.delete_mode
            return
        
        # Click en pistas
        for button in self.track_buttons:
            # Ajustar rect para scroll
            adjusted_rect = button['rect'].copy()
            adjusted_rect.y -= self.scroll_offset
            
            if adjusted_rect.collidepoint(pos):
                if self.delete_mode:
                    # Modo eliminar
                    self.delete_track(button['track'])
                else:
                    # Seleccionar pista
                    self.selected_track = button['track']
                return
    
    def handle_scroll(self, y):
        """Maneja el scroll del mouse"""
        self.scroll_offset = max(0, min(self.scroll_offset - y * 30, self.max_scroll))
    
    def delete_track(self, track):
        """Elimina una pista"""
        try:
            track_name = track['name']
            
            # Confirmar eliminación
            print(f"Eliminando pista: {track_name}")
            
            # Eliminar archivos
            files_to_delete = [
                f"tracks/{track_name}.png",
                f"tracks/{track_name}.json",
                f"tracks/{track_name}_thumb.png"
            ]
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Recargar pistas
            self.load_tracks()
            self.create_ui()
            
            # Deseleccionar si era la pista seleccionada
            if self.selected_track and self.selected_track['name'] == track_name:
                self.selected_track = None
            
            print(f"✓ Pista eliminada: {track_name}")
            
        except Exception as e:
            print(f"Error eliminando pista: {e}")
    
    def launch_track_editor(self):
        """Lanza el editor de pistas"""
        print("Lanzando editor de pistas...")
        pygame.quit()
        
        # Importar y ejecutar el editor
        try:
            import track_editor
            editor = track_editor.TrackEditor()
            editor.run()
        except Exception as e:
            print(f"Error lanzando editor: {e}")
        
        # Reiniciar selector después de cerrar editor
        self.__init__(self.width, self.height)
    
    def draw(self):
        """Dibuja la interfaz"""
        self.screen.fill(DARK_GRAY)
        
        # Título
        title = self.title_font.render("Selector de Pistas", True, YELLOW)
        title_rect = title.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title, title_rect)
        
        # Instrucciones
        if self.delete_mode:
            instruction = self.small_font.render("MODO ELIMINAR: Click en una pista para eliminarla", True, RED)
        else:
            instruction = self.small_font.render("Selecciona una pista para entrenar", True, WHITE)
        inst_rect = instruction.get_rect(center=(self.width // 2, 90))
        self.screen.blit(instruction, inst_rect)
        
        # Dibujar pistas
        for button in self.track_buttons:
            # Ajustar posición por scroll
            adjusted_rect = button['rect'].copy()
            adjusted_rect.y -= self.scroll_offset
            
            # Solo dibujar si está visible
            if adjusted_rect.bottom < 0 or adjusted_rect.top > self.height - 120:
                continue
            
            track = button['track']
            
            # Fondo del botón
            is_selected = (self.selected_track and 
                          self.selected_track['name'] == track['name'])
            
            border_color = YELLOW if is_selected else WHITE
            border_width = 4 if is_selected else 2
            
            pygame.draw.rect(self.screen, BLACK, adjusted_rect)
            pygame.draw.rect(self.screen, border_color, adjusted_rect, border_width)
            
            # Thumbnail
            thumb_rect = button['thumbnail_rect'].copy()
            thumb_rect.y -= self.scroll_offset
            self.screen.blit(track['thumbnail'], thumb_rect)
            
            # Nombre de la pista
            name_text = self.small_font.render(track['name'], True, WHITE)
            name_rect = name_text.get_rect(
                centerx=adjusted_rect.centerx,
                top=thumb_rect.bottom + 5
            )
            self.screen.blit(name_text, name_rect)
            
            # Fecha de creación
            created = track['metadata'].get('created', 'N/A')
            if created != 'N/A':
                try:
                    date_obj = datetime.strptime(created, "%Y%m%d_%H%M%S")
                    date_str = date_obj.strftime("%d/%m/%Y %H:%M")
                except:
                    date_str = created
            else:
                date_str = 'N/A'
            
            date_text = self.small_font.render(date_str, True, LIGHT_GRAY)
            date_rect = date_text.get_rect(
                centerx=adjusted_rect.centerx,
                top=name_rect.bottom + 2
            )
            self.screen.blit(date_text, date_rect)
            
            # Botón eliminar (X roja en esquina)
            if self.delete_mode:
                delete_rect = button['delete_rect'].copy()
                delete_rect.y -= self.scroll_offset
                pygame.draw.circle(self.screen, RED, delete_rect.center, 15)
                pygame.draw.circle(self.screen, WHITE, delete_rect.center, 15, 2)
                
                # X
                x_font = pygame.font.Font(None, 24)
                x_text = x_font.render("X", True, WHITE)
                x_rect = x_text.get_rect(center=delete_rect.center)
                self.screen.blit(x_text, x_rect)
        
        # Barra de scroll (si es necesaria)
        if self.max_scroll > 0:
            scroll_bar_height = max(50, (self.height - 200) * (self.height - 200) / (self.height - 200 + self.max_scroll))
            scroll_bar_y = 120 + (self.scroll_offset / self.max_scroll) * (self.height - 200 - scroll_bar_height)
            
            pygame.draw.rect(self.screen, GRAY, 
                           (self.width - 20, 120, 15, self.height - 200))
            pygame.draw.rect(self.screen, WHITE,
                           (self.width - 20, scroll_bar_y, 15, scroll_bar_height))
        
        # Botones inferiores
        self.draw_button(self.create_new_button)
        self.draw_button(self.start_training_button, 
                        enabled=(self.selected_track is not None))
        
        # Botón modo eliminar
        delete_btn = self.delete_button.copy()
        if self.delete_mode:
            delete_btn['color'] = ORANGE
            delete_btn['label'] = 'Modo Normal'
        self.draw_button(delete_btn)
        
        # Mensaje si no hay pistas
        if len(self.tracks) == 0:
            no_tracks = self.font.render("No hay pistas guardadas", True, LIGHT_GRAY)
            no_tracks_rect = no_tracks.get_rect(center=(self.width // 2, self.height // 2 - 50))
            self.screen.blit(no_tracks, no_tracks_rect)
            
            hint = self.small_font.render("Haz click en 'Crear Nueva Pista' para empezar", True, LIGHT_GRAY)
            hint_rect = hint.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(hint, hint_rect)
    
    def draw_button(self, button, enabled=True):
        """Dibuja un botón"""
        color = button['color'] if enabled else GRAY
        text_color = WHITE if enabled else DARK_GRAY
        
        # Fondo
        pygame.draw.rect(self.screen, color, button['rect'])
        pygame.draw.rect(self.screen, WHITE, button['rect'], 2)
        
        # Texto
        text = self.small_font.render(button['label'], True, text_color)
        text_rect = text.get_rect(center=button['rect'].center)
        self.screen.blit(text, text_rect)
    
    def run(self):
        """Loop principal del selector"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.selected_track = None
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.selected_track = None
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Click izquierdo
                        self.handle_click(event.pos)
                    elif event.button == 4:  # Scroll up
                        self.handle_scroll(1)
                    elif event.button == 5:  # Scroll down
                        self.handle_scroll(-1)
                
                elif event.type == pygame.MOUSEWHEEL:
                    self.handle_scroll(event.y)
            
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        return self.selected_track


def select_track():
    """Función helper para seleccionar una pista"""
    selector = TrackSelector()
    selected = selector.run()
    return selected


if __name__ == "__main__":
    selected_track = select_track()
    if selected_track:
        print(f"\n✓ Pista seleccionada: {selected_track['name']}")
    else:
        print("\n✗ No se seleccionó ninguna pista")
