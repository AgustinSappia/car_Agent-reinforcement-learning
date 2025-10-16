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
        
        # Botones
        self.create_new_button = None
        self.start_training_button = None
        self.delete_mode = False
        
        # Cargar pistas
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
        
        track_files = [f for f in os.listdir('tracks') if f.endswith('.json')]
        
        for json_file in track_files:
            try:
                with open(f"tracks/{json_file}", 'r') as f:
                    metadata = json.load(f)
                    
                    # Nombre base (sin extensión)
                    track_name = metadata.get('name', json_file.replace('.json', ''))
                    track_base_path = f"tracks/{track_name}"
                    
                    # Rutas corregidas
                    track_path = f"{track_base_path}_track.png"
                    thumb_path = f"{track_base_path}_thumb.png"
                    
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
                            'path': track_base_path  # ← Base sin extensión
                        })
            except Exception as e:
                print(f"Error cargando pista {json_file}: {e}")
        
        # Ordenar por fecha
        self.tracks.sort(key=lambda x: x['metadata'].get('created', ''), reverse=True)
    
    def create_ui(self):
        """Crea botones y grid de pistas"""
        self.create_new_button = {
            'rect': pygame.Rect(50, self.height - 80, 250, 60),
            'label': 'Crear Nueva Pista',
            'color': GREEN
        }
        self.start_training_button = {
            'rect': pygame.Rect(self.width - 300, self.height - 80, 250, 60),
            'label': 'Iniciar Entrenamiento',
            'color': BLUE
        }
        self.delete_button = {
            'rect': pygame.Rect(self.width // 2 - 125, self.height - 80, 250, 60),
            'label': 'Modo Eliminar',
            'color': RED
        }
        
        cols = 4
        thumb_w, thumb_h, pad = 200, 150, 30
        start_y = 120
        self.track_buttons = []
        
        for i, track in enumerate(self.tracks):
            row, col = divmod(i, cols)
            x = 50 + col * (thumb_w + pad)
            y = start_y + row * (thumb_h + pad + 60)
            
            self.track_buttons.append({
                'rect': pygame.Rect(x, y, thumb_w, thumb_h + 60),
                'track': track,
                'thumbnail_rect': pygame.Rect(x, y, thumb_w, thumb_h),
                'delete_rect': pygame.Rect(x + thumb_w - 30, y, 30, 30)
            })
        
        if self.track_buttons:
            last_button = self.track_buttons[-1]
            self.max_scroll = max(0, last_button['rect'].bottom - (self.height - 120))
    
    def handle_click(self, pos):
        adjusted_pos = (pos[0], pos[1] + self.scroll_offset)
        
        if self.create_new_button['rect'].collidepoint(pos):
            self.launch_track_editor()
            return
        if self.start_training_button['rect'].collidepoint(pos):
            if self.selected_track:
                self.running = False
            return
        if self.delete_button['rect'].collidepoint(pos):
            self.delete_mode = not self.delete_mode
            return
        
        for button in self.track_buttons:
            adjusted_rect = button['rect'].copy()
            adjusted_rect.y -= self.scroll_offset
            if adjusted_rect.collidepoint(pos):
                if self.delete_mode:
                    self.delete_track(button['track'])
                else:
                    self.selected_track = button['track']
                return
    
    def handle_scroll(self, y):
        self.scroll_offset = max(0, min(self.scroll_offset - y * 30, self.max_scroll))
    
    def delete_track(self, track):
        """Elimina los archivos relacionados a una pista"""
        try:
            name = track['name']
            print(f"Eliminando pista: {name}")
            files = [
                f"tracks/{name}.json",
                f"tracks/{name}_track.png",
                f"tracks/{name}_thumb.png",
                f"tracks/{name}_checkpoint.png",
                f"tracks/{name}_finish.png",
                f"tracks/{name}_speed.png",
                f"tracks/{name}_slow.png",
            ]
            for fpath in files:
                if os.path.exists(fpath):
                    os.remove(fpath)
            self.load_tracks()
            self.create_ui()
            if self.selected_track and self.selected_track['name'] == name:
                self.selected_track = None
            print(f"✓ Pista eliminada: {name}")
        except Exception as e:
            print(f"Error eliminando pista: {e}")
    
    def launch_track_editor(self):
        """Abre el editor de pistas"""
        print("Lanzando editor de pistas...")
        pygame.quit()
        try:
            import track_editor_v2
            editor = track_editor_v2.TrackEditorV2()
            editor.run()
        except Exception as e:
            print(f"Error lanzando editor: {e}")
        self.__init__(self.width, self.height)
    
    def draw_button(self, button, enabled=True):
        color = button['color'] if enabled else GRAY
        pygame.draw.rect(self.screen, color, button['rect'])
        pygame.draw.rect(self.screen, WHITE, button['rect'], 2)
        text = self.small_font.render(button['label'], True, WHITE)
        text_rect = text.get_rect(center=button['rect'].center)
        self.screen.blit(text, text_rect)
    
    def draw(self):
        self.screen.fill(DARK_GRAY)
        title = self.title_font.render("Selector de Pistas", True, YELLOW)
        self.screen.blit(title, title.get_rect(center=(self.width//2, 50)))
        
        for button in self.track_buttons:
            rect = button['rect'].copy()
            rect.y -= self.scroll_offset
            if rect.bottom < 0 or rect.top > self.height - 120:
                continue
            track = button['track']
            selected = self.selected_track and self.selected_track['name'] == track['name']
            border_color = YELLOW if selected else WHITE
            pygame.draw.rect(self.screen, BLACK, rect)
            pygame.draw.rect(self.screen, border_color, rect, 4 if selected else 2)
            thumb_rect = button['thumbnail_rect'].copy()
            thumb_rect.y -= self.scroll_offset
            self.screen.blit(track['thumbnail'], thumb_rect)
            name_text = self.small_font.render(track['name'], True, WHITE)
            name_rect = name_text.get_rect(centerx=rect.centerx, top=thumb_rect.bottom + 5)
            self.screen.blit(name_text, name_rect)
        
        self.draw_button(self.create_new_button)
        self.draw_button(self.start_training_button, enabled=(self.selected_track is not None))
        delete_btn = self.delete_button.copy()
        if self.delete_mode:
            delete_btn['color'] = ORANGE
            delete_btn['label'] = 'Modo Normal'
        self.draw_button(delete_btn)
        pygame.display.flip()
    
    def run(self):
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                    self.selected_track = None
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    self.running = False
                    self.selected_track = None
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 1:
                        self.handle_click(e.pos)
                    elif e.button == 4:
                        self.handle_scroll(1)
                    elif e.button == 5:
                        self.handle_scroll(-1)
            self.draw()
            self.clock.tick(60)
        pygame.quit()
        return self.selected_track


def select_track():
    selector = TrackSelector()
    return selector.run()


if __name__ == "__main__":
    track = select_track()
    if track:
        print(f"✓ Pista seleccionada: {track['name']}")
    else:
        print("✗ No se seleccionó ninguna pista")
