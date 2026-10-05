"""
Selector de pistas estilo videojuego.

- Grilla de pistas con miniaturas + tarjeta "Nueva pista"
- Panel de detalle con vista previa grande y datos de la pista
- Acciones: Entrenar, Editar, Duplicar, Renombrar, Eliminar
- Eliminar pide confirmación y manda la pista a una papelera (se puede deshacer)
- Se maneja con mouse o con teclado
"""

import pygame
import json
import os
import shutil
from datetime import datetime

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG = (24, 26, 34)
PANEL = (36, 39, 50)
PANEL_LIGHT = (52, 56, 70)
TEXT_DIM = (150, 155, 170)
YELLOW = (255, 205, 40)
GREEN = (60, 200, 110)
BLUE = (60, 130, 255)
RED = (230, 70, 70)
ORANGE = (255, 150, 40)
CYAN = (70, 210, 230)

TRACKS_DIR = 'tracks'
TRASH_DIR = os.path.join(TRACKS_DIR, '_papelera')
TRACK_SUFFIXES = ['.json', '.png', '_track.png', '_thumb.png', '_checkpoint.png',
                  '_finish.png', '_speed.png', '_slow.png']

NEW_TRACK = 'NEW'  # Marcador de la tarjeta "Nueva pista"


def track_files(name, folder=TRACKS_DIR):
    """Archivos que forman una pista (solo los que existen)"""
    paths = [os.path.join(folder, f"{name}{s}") for s in TRACK_SUFFIXES]
    return [p for p in paths if os.path.exists(p)]


def display_name(metadata, fallback):
    """Nombre visible: el que puso el usuario o la fecha de creación"""
    if metadata.get('display_name'):
        return metadata['display_name']
    created = metadata.get('created', '')
    try:
        return "Pista " + datetime.strptime(created, "%Y%m%d_%H%M%S").strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return fallback


def format_date(stamp):
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%d/%m/%Y %H:%M")
    except (ValueError, TypeError):
        return "-"


def fit_image(img, size):
    """Escala manteniendo proporción y centra sobre un fondo negro"""
    w, h = size
    iw, ih = img.get_size()
    scale = min(w / iw, h / ih)
    scaled = pygame.transform.smoothscale(img, (max(1, int(iw * scale)), max(1, int(ih * scale))))
    surf = pygame.Surface(size)
    surf.fill(BLACK)
    surf.blit(scaled, scaled.get_rect(center=(w // 2, h // 2)))
    return surf


class TrackSelector:
    """Selector gráfico de pistas con miniaturas"""

    # Layout
    GRID_X, GRID_Y = 30, 95
    GRID_W, GRID_H = 780, 545
    COLS = 3
    CARD_W, CARD_H = 240, 200
    THUMB_H = 160
    GAP = 25
    PANEL_X = 840

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.init_display()

        # Estado
        self.running = True
        self.result = None          # Pista elegida para entrenar
        self.tracks = []
        self.selected = 0           # Índice en self.items (0 = "Nueva pista")
        self.hover = None
        self.scroll = 0
        self.modal = None           # None | 'delete' | 'rename'
        self.rename_text = ''
        self.toast = None           # (texto, color, expira_ms, puede_deshacer)
        self.undo_stack = []        # Nombres de pistas en la papelera (para deshacer)
        self.last_click = (None, 0)
        self.preview_cache = {}

        self.load_tracks()
        if self.tracks:
            self.selected = 1

        print("\n" + "=" * 60)
        print("SELECTOR DE PISTAS")
        print("=" * 60)
        print(f"Pistas disponibles: {len(self.tracks)}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # Inicialización / datos
    # ------------------------------------------------------------------ #
    def init_display(self):
        """(Re)abre la ventana. Se llama de nuevo al volver del editor."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Selector de Pistas - Self Driving Car AI")
        self.clock = pygame.time.Clock()
        pygame.key.set_repeat(300, 60)
        self.title_font = pygame.font.Font(None, 52)
        self.big_font = pygame.font.Font(None, 38)
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 24)
        self.tiny_font = pygame.font.Font(None, 20)

    def load_tracks(self):
        """Carga todas las pistas disponibles"""
        self.tracks = []
        self.preview_cache = {}
        os.makedirs(TRACKS_DIR, exist_ok=True)

        for json_file in sorted(os.listdir(TRACKS_DIR)):
            if not json_file.endswith('.json'):
                continue
            try:
                with open(os.path.join(TRACKS_DIR, json_file), 'r') as f:
                    metadata = json.load(f)
                track_name = metadata.get('name', json_file[:-5])
                track_base_path = f"{TRACKS_DIR}/{track_name}"
                if not os.path.exists(f"{track_base_path}_track.png"):
                    continue

                # La imagen combinada muestra zonas, meta y checkpoints
                img_path = f"{track_base_path}.png"
                if not os.path.exists(img_path):
                    img_path = f"{track_base_path}_track.png"
                image = pygame.image.load(img_path)

                self.tracks.append({
                    'name': track_name,
                    'metadata': metadata,
                    'thumbnail': fit_image(image, (self.CARD_W, self.THUMB_H)),
                    'image': image,
                    'path': track_base_path,
                    'title': display_name(metadata, track_name),
                })
            except Exception as e:
                print(f"Error cargando pista {json_file}: {e}")

        # Más nuevas primero
        self.tracks.sort(key=lambda t: t['metadata'].get('created', ''), reverse=True)

    @property
    def items(self):
        return [NEW_TRACK] + self.tracks

    @property
    def current(self):
        """Pista seleccionada (None si está seleccionada la tarjeta 'Nueva')"""
        if 0 < self.selected < len(self.items):
            return self.items[self.selected]
        return None

    def select_by_name(self, name):
        for i, t in enumerate(self.tracks):
            if t['name'] == name:
                self.selected = i + 1
                self.ensure_visible()
                return
        self.selected = min(self.selected, len(self.items) - 1)

    @staticmethod
    def problems(track):
        """Lista de cosas que le faltan a la pista para entrenar bien"""
        m = track['metadata']
        issues = []
        if not m.get('spawn_point'):
            issues.append("Falta punto de salida")
        if not m.get('finish_line'):
            issues.append("Falta línea de meta")
        if not m.get('checkpoints'):
            issues.append("Sin checkpoints")
        return issues

    # ------------------------------------------------------------------ #
    # Acciones
    # ------------------------------------------------------------------ #
    def start_training(self):
        track = self.current
        if track is None:
            self.launch_track_editor()
            return
        self.result = track
        self.running = False

    def launch_track_editor(self, track_name=None):
        """Abre el editor (pista nueva o una existente) y vuelve al selector"""
        print(f"Lanzando editor de pistas... {'(editando ' + track_name + ')' if track_name else ''}")
        before = {t['name'] for t in self.tracks}
        pygame.quit()
        try:
            import track_editor_v3
            editor = track_editor_v3.TrackEditorV3(track_name=track_name)
            editor.run()
        except Exception as e:
            print(f"Error lanzando editor: {e}")
        self.init_display()
        self.load_tracks()

        new = [t['name'] for t in self.tracks if t['name'] not in before]
        if new:
            self.select_by_name(new[0])
            self.show_toast("Pista nueva guardada", GREEN)
        elif track_name:
            self.select_by_name(track_name)

    def duplicate_track(self):
        track = self.current
        if not track:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"track_{stamp}"
        if os.path.exists(os.path.join(TRACKS_DIR, f"{new_name}.json")):
            self.show_toast("Esperá un segundo antes de duplicar otra vez", ORANGE)
            return
        for path in track_files(track['name']):
            suffix = os.path.basename(path)[len(track['name']):]
            if suffix == '.json':
                continue
            shutil.copy2(path, os.path.join(TRACKS_DIR, f"{new_name}{suffix}"))
        meta = dict(track['metadata'])
        meta.update({'name': new_name, 'created': stamp, 'modified': stamp,
                     'display_name': f"{track['title']} (copia)"})
        with open(os.path.join(TRACKS_DIR, f"{new_name}.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        self.load_tracks()
        self.select_by_name(new_name)
        self.show_toast("Pista duplicada", GREEN)

    def rename_track(self, new_title):
        track = self.current
        new_title = new_title.strip()
        if not track or not new_title:
            return
        meta = dict(track['metadata'])
        meta['display_name'] = new_title
        with open(os.path.join(TRACKS_DIR, f"{track['name']}.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        name = track['name']
        self.load_tracks()
        self.select_by_name(name)
        self.show_toast("Nombre actualizado", GREEN)

    def delete_track(self, track):
        """Mueve la pista a la papelera (se puede deshacer)"""
        try:
            name = track['name']
            dest = os.path.join(TRASH_DIR, name)
            os.makedirs(dest, exist_ok=True)
            for path in track_files(name):
                shutil.move(path, os.path.join(dest, os.path.basename(path)))
            self.undo_stack.append(name)
            index = self.selected
            self.load_tracks()
            self.selected = max(0, min(index, len(self.items) - 1))
            self.ensure_visible()
            self.show_toast(f"\"{track['title']}\" eliminada", RED, can_undo=True)
            print(f"✓ Pista movida a la papelera: {name}")
        except Exception as e:
            print(f"Error eliminando pista: {e}")
            self.show_toast("No se pudo eliminar la pista", RED)

    def undo_delete(self):
        if not self.undo_stack:
            return
        name = self.undo_stack.pop()
        src = os.path.join(TRASH_DIR, name)
        if os.path.isdir(src):
            for fname in os.listdir(src):
                shutil.move(os.path.join(src, fname), os.path.join(TRACKS_DIR, fname))
            os.rmdir(src)
        self.load_tracks()
        self.select_by_name(name)
        self.show_toast("Pista restaurada", GREEN)

    def show_toast(self, text, color, can_undo=False):
        self.toast = (text, color, pygame.time.get_ticks() + 6000, can_undo)

    # ------------------------------------------------------------------ #
    # Layout helpers
    # ------------------------------------------------------------------ #
    def card_rect(self, i):
        """Rect de la tarjeta i en coordenadas de pantalla (con scroll)"""
        row, col = divmod(i, self.COLS)
        x = self.GRID_X + col * (self.CARD_W + self.GAP)
        y = self.GRID_Y + row * (self.CARD_H + self.GAP) - self.scroll
        return pygame.Rect(x, y, self.CARD_W, self.CARD_H)

    def max_scroll(self):
        rows = (len(self.items) + self.COLS - 1) // self.COLS
        content = rows * (self.CARD_H + self.GAP) - self.GAP
        return max(0, content - self.GRID_H)

    def ensure_visible(self):
        r = self.card_rect(self.selected)
        if r.top < self.GRID_Y:
            self.scroll -= self.GRID_Y - r.top
        elif r.bottom > self.GRID_Y + self.GRID_H:
            self.scroll += r.bottom - (self.GRID_Y + self.GRID_H)
        self.scroll = max(0, min(self.scroll, self.max_scroll()))

    def grid_area(self):
        return pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_W, self.GRID_H)

    def card_at(self, pos):
        if not self.grid_area().collidepoint(pos):
            return None
        for i in range(len(self.items)):
            if self.card_rect(i).collidepoint(pos):
                return i
        return None

    def action_buttons(self):
        """Botones del panel de detalle: (id, texto, color, rect, habilitado)"""
        x, w = self.PANEL_X + 20, self.width - self.PANEL_X - 50
        has_track = self.current is not None
        buttons = [('train', "ENTRENAR  (Enter)", GREEN, pygame.Rect(x, 500, w, 52), has_track)]
        small_w = (w - 10) // 2
        row = [('edit', "Editar (E)", BLUE, True), ('duplicate', "Duplicar (D)", PANEL_LIGHT, True),
               ('rename', "Renombrar (R)", PANEL_LIGHT, True), ('delete', "Eliminar (Supr)", RED, True)]
        for k, (bid, label, color, _) in enumerate(row):
            bx = x + (k % 2) * (small_w + 10)
            by = 562 + (k // 2) * 46
            buttons.append((bid, label, color, pygame.Rect(bx, by, small_w, 38), has_track))
        if not has_track:
            buttons = [('new', "CREAR PISTA  (Enter)", GREEN, pygame.Rect(x, 500, w, 52), True)]
        return buttons

    def modal_buttons(self):
        cx, cy = self.width // 2, self.height // 2
        if self.modal == 'delete':
            return [('cancel', "Cancelar (Esc)", PANEL_LIGHT, pygame.Rect(cx - 210, cy + 125, 200, 46)),
                    ('confirm', "Eliminar (Enter)", RED, pygame.Rect(cx + 10, cy + 125, 200, 46))]
        return [('cancel', "Cancelar (Esc)", PANEL_LIGHT, pygame.Rect(cx - 210, cy + 50, 200, 46)),
                ('confirm', "Guardar (Enter)", GREEN, pygame.Rect(cx + 10, cy + 50, 200, 46))]

    def toast_undo_rect(self):
        return pygame.Rect(self.width - 160, 20, 130, 40)

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    def do_action(self, action):
        if action in ('train', 'new'):
            self.start_training()
        elif self.current is None:
            return
        elif action == 'edit':
            self.launch_track_editor(self.current['name'])
        elif action == 'duplicate':
            self.duplicate_track()
        elif action == 'rename':
            self.modal = 'rename'
            self.rename_text = self.current['title']
        elif action == 'delete':
            self.modal = 'delete'

    def close_modal(self, confirm):
        modal, self.modal = self.modal, None
        if not confirm:
            return
        if modal == 'delete' and self.current:
            self.delete_track(self.current)
        elif modal == 'rename':
            self.rename_track(self.rename_text)

    def handle_click(self, pos):
        if self.modal:
            for bid, _, _, rect in self.modal_buttons():
                if rect.collidepoint(pos):
                    self.close_modal(bid == 'confirm')
            return

        if self.toast and self.toast[3] and self.toast_undo_rect().collidepoint(pos):
            self.undo_delete()
            return

        for bid, _, _, rect, enabled in self.action_buttons():
            if enabled and rect.collidepoint(pos):
                self.do_action(bid)
                return

        i = self.card_at(pos)
        if i is None:
            return
        now = pygame.time.get_ticks()
        double = self.last_click[0] == i and now - self.last_click[1] < 400
        self.last_click = (i, now)
        self.selected = i
        if double or i == 0:
            self.start_training()

    def handle_key(self, event):
        key = event.key
        if self.modal == 'rename':
            if key == pygame.K_RETURN:
                self.close_modal(True)
            elif key == pygame.K_ESCAPE:
                self.close_modal(False)
            elif key == pygame.K_BACKSPACE:
                self.rename_text = self.rename_text[:-1]
            elif event.unicode and event.unicode.isprintable() and len(self.rename_text) < 30:
                self.rename_text += event.unicode
            return
        if self.modal == 'delete':
            if key in (pygame.K_RETURN, pygame.K_y, pygame.K_s):
                self.close_modal(True)
            elif key in (pygame.K_ESCAPE, pygame.K_n):
                self.close_modal(False)
            return

        n = len(self.items)
        if key == pygame.K_ESCAPE:
            self.result = None
            self.running = False
        elif key == pygame.K_RIGHT:
            self.selected = min(n - 1, self.selected + 1)
        elif key == pygame.K_LEFT:
            self.selected = max(0, self.selected - 1)
        elif key == pygame.K_DOWN:
            self.selected = min(n - 1, self.selected + self.COLS)
        elif key == pygame.K_UP:
            self.selected = max(0, self.selected - self.COLS)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.start_training()
        elif key == pygame.K_n:
            self.launch_track_editor()
        elif key == pygame.K_e:
            self.do_action('edit')
        elif key == pygame.K_d:
            self.do_action('duplicate')
        elif key == pygame.K_r:
            self.do_action('rename')
        elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            self.do_action('delete')
        elif key == pygame.K_z and event.mod & pygame.KMOD_CTRL:
            self.undo_delete()
        self.ensure_visible()

    def handle_scroll(self, y):
        self.scroll = max(0, min(self.scroll - y * 40, self.max_scroll()))

    # ------------------------------------------------------------------ #
    # Dibujo
    # ------------------------------------------------------------------ #
    def text(self, txt, font, color, **pos):
        surf = font.render(txt, True, color)
        rect = surf.get_rect(**pos)
        self.screen.blit(surf, rect)
        return rect

    def fit_text(self, txt, font, max_w):
        if font.size(txt)[0] <= max_w:
            return txt
        while txt and font.size(txt + "...")[0] > max_w:
            txt = txt[:-1]
        return txt + "..."

    def draw_button(self, label, color, rect, enabled=True, hovered=False):
        base = color if enabled else PANEL_LIGHT
        if hovered and enabled:
            base = tuple(min(255, c + 30) for c in base)
        pygame.draw.rect(self.screen, base, rect, border_radius=8)
        if hovered and enabled:
            pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=8)
        self.text(label, self.small_font, WHITE if enabled else TEXT_DIM, center=rect.center)

    def draw_card(self, i, item):
        rect = self.card_rect(i)
        selected = i == self.selected
        hovered = i == self.hover
        pygame.draw.rect(self.screen, PANEL_LIGHT if hovered else PANEL, rect, border_radius=10)

        if item == NEW_TRACK:
            plus = pygame.Rect(0, 0, 70, 70)
            plus.center = (rect.centerx, rect.top + self.THUMB_H // 2)
            pygame.draw.rect(self.screen, GREEN, plus, 4, border_radius=35)
            pygame.draw.line(self.screen, GREEN, (plus.centerx, plus.top + 18), (plus.centerx, plus.bottom - 18), 5)
            pygame.draw.line(self.screen, GREEN, (plus.left + 18, plus.centery), (plus.right - 18, plus.centery), 5)
            self.text("Nueva pista  (N)", self.font, GREEN, center=(rect.centerx, rect.top + self.THUMB_H + 20))
        else:
            thumb_rect = pygame.Rect(rect.x, rect.y, self.CARD_W, self.THUMB_H)
            self.screen.blit(item['thumbnail'], thumb_rect)
            title = self.fit_text(item['title'], self.small_font, self.CARD_W - 20)
            self.text(title, self.small_font, WHITE, center=(rect.centerx, rect.top + self.THUMB_H + 20))
            if self.problems(item):
                badge = pygame.Rect(rect.right - 30, rect.top + 8, 22, 22)
                pygame.draw.circle(self.screen, ORANGE, badge.center, 11)
                self.text("!", self.small_font, BLACK, center=badge.center)

        if selected:
            color = RED if self.modal == 'delete' else YELLOW
            pygame.draw.rect(self.screen, color, rect.inflate(8, 8), 4, border_radius=12)

    def draw_detail_panel(self):
        panel = pygame.Rect(self.PANEL_X, 95, self.width - self.PANEL_X - 30, 600)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=12)
        x = panel.x + 20
        track = self.current

        if track is None:
            self.text("Crear una pista nueva", self.big_font, GREEN, topleft=(x, panel.y + 20))
            lines = ["Dibujá el circuito, poné la salida,",
                     "la meta y los checkpoints.",
                     "Después la elegís acá para entrenar",
                     "a los autos."]
            for k, line in enumerate(lines):
                self.text(line, self.small_font, TEXT_DIM, topleft=(x, panel.y + 75 + k * 28))
        else:
            preview_size = (panel.w - 40, 220)
            key = track['name']
            if key not in self.preview_cache:
                self.preview_cache[key] = fit_image(track['image'], preview_size)
            self.screen.blit(self.preview_cache[key], (x, panel.y + 20))
            pygame.draw.rect(self.screen, PANEL_LIGHT, (x, panel.y + 20, *preview_size), 2)

            y = panel.y + 255
            self.text(self.fit_text(track['title'], self.big_font, panel.w - 40), self.big_font, YELLOW, topleft=(x, y))
            m = track['metadata']
            info = [
                ("Vueltas", str(m.get('required_laps', 1))),
                ("Checkpoints", str(len(m.get('checkpoints', [])))),
                ("Salida", "Sí" if m.get('spawn_point') else "No"),
                ("Meta", "Sí" if m.get('finish_line') else "No"),
                ("Creada", format_date(m.get('created'))),
            ]
            for k, (label, value) in enumerate(info):
                col_x = x + (k % 2) * 190
                row_y = y + 42 + (k // 2) * 26
                self.text(f"{label}:", self.small_font, TEXT_DIM, topleft=(col_x, row_y))
                self.text(value, self.small_font, WHITE, topleft=(col_x + 125, row_y))

            issues = self.problems(track)
            if issues:
                self.text("! " + ", ".join(issues), self.tiny_font, ORANGE, topleft=(x, y + 120))

        mouse = pygame.mouse.get_pos()
        for _, label, color, rect, enabled in self.action_buttons():
            self.draw_button(label, color, rect, enabled, rect.collidepoint(mouse) and not self.modal)

    def draw_modal(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))
        cx, cy = self.width // 2, self.height // 2
        track = self.current
        mouse = pygame.mouse.get_pos()

        if self.modal == 'delete':
            box = pygame.Rect(0, 0, 520, 380)
            box.center = (cx, cy)
            pygame.draw.rect(self.screen, PANEL, box, border_radius=14)
            pygame.draw.rect(self.screen, RED, box, 3, border_radius=14)
            self.text("¿Eliminar esta pista?", self.big_font, RED, center=(cx, box.top + 35))
            thumb = track['thumbnail']
            self.screen.blit(thumb, thumb.get_rect(center=(cx, box.top + 145)))
            self.text(self.fit_text(track['title'], self.font, 480), self.font, WHITE, center=(cx, box.top + 255))
            self.text("Podés deshacerlo con Ctrl+Z", self.tiny_font, TEXT_DIM, center=(cx, box.top + 283))
        else:
            box = pygame.Rect(0, 0, 520, 220)
            box.center = (cx, cy)
            pygame.draw.rect(self.screen, PANEL, box, border_radius=14)
            pygame.draw.rect(self.screen, YELLOW, box, 3, border_radius=14)
            self.text("Renombrar pista", self.big_font, YELLOW, center=(cx, box.top + 35))
            field = pygame.Rect(box.left + 30, box.top + 70, box.w - 60, 46)
            pygame.draw.rect(self.screen, BG, field, border_radius=6)
            pygame.draw.rect(self.screen, WHITE, field, 2, border_radius=6)
            cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            self.text(self.rename_text + cursor, self.font, WHITE, midleft=(field.left + 12, field.centery))

        for _, label, color, rect in self.modal_buttons():
            self.draw_button(label, color, rect, True, rect.collidepoint(mouse))

    def draw_toast(self):
        if not self.toast:
            return
        text, color, expires, can_undo = self.toast
        if pygame.time.get_ticks() > expires:
            self.toast = None
            return
        width = self.small_font.size(text)[0] + 40 + (150 if can_undo else 0)
        box = pygame.Rect(self.width - width - 20, 15, width, 50)
        pygame.draw.rect(self.screen, PANEL_LIGHT, box, border_radius=10)
        pygame.draw.rect(self.screen, color, box, 2, border_radius=10)
        self.text(text, self.small_font, WHITE, midleft=(box.left + 20, box.centery))
        if can_undo:
            r = self.toast_undo_rect()
            r.centery = box.centery
            self.draw_button("Deshacer", BLUE, r, True, r.collidepoint(pygame.mouse.get_pos()))

    def draw(self):
        self.screen.fill(BG)
        self.text("SELECCIONÁ UNA PISTA", self.title_font, YELLOW, topleft=(self.GRID_X, 30))
        self.text(f"{len(self.tracks)} pistas", self.small_font, TEXT_DIM, topleft=(self.GRID_X + 470, 45))

        self.screen.set_clip(self.grid_area().inflate(10, 10))
        for i, item in enumerate(self.items):
            if self.card_rect(i).colliderect(self.grid_area().inflate(10, 10)):
                self.draw_card(i, item)
        self.screen.set_clip(None)

        # Barra de scroll
        if self.max_scroll() > 0:
            area = self.grid_area()
            bar_h = max(40, area.h * area.h // (area.h + self.max_scroll()))
            bar_y = area.top + (area.h - bar_h) * self.scroll // self.max_scroll()
            pygame.draw.rect(self.screen, PANEL_LIGHT, (area.right + 4, bar_y, 6, bar_h), border_radius=3)

        self.draw_detail_panel()

        hints = "Flechas: moverse   Enter / doble clic: entrenar   E: editar   D: duplicar   R: renombrar   Supr: eliminar   Esc: salir"
        self.text(hints, self.tiny_font, TEXT_DIM, midbottom=(self.GRID_X + self.GRID_W // 2, self.height - 15))

        if self.modal:
            self.draw_modal()
        self.draw_toast()
        pygame.display.flip()

    # ------------------------------------------------------------------ #
    def run(self):
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.result = None
                    self.running = False
                elif e.type == pygame.KEYDOWN:
                    self.handle_key(e)
                elif e.type == pygame.MOUSEMOTION:
                    self.hover = None if self.modal else self.card_at(e.pos)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.handle_click(e.pos)
                elif e.type == pygame.MOUSEWHEEL and not self.modal:
                    self.handle_scroll(e.y)
                if not self.running:
                    break
            if not self.running:
                break
            self.draw()
            self.clock.tick(60)
        pygame.quit()
        return self.result


def select_track():
    selector = TrackSelector()
    return selector.run()


if __name__ == "__main__":
    track = select_track()
    if track:
        print(f"✓ Pista seleccionada: {track['name']}")
    else:
        print("✗ No se seleccionó ninguna pista")
