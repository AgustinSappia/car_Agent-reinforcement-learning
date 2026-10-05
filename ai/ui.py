"""Widgets simples de pygame compartidos por los menús y la pantalla de entrenamiento"""

import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG = (24, 26, 34)
PANEL = (36, 39, 50)
PANEL_LIGHT = (52, 56, 70)
TEXT_DIM = (150, 155, 170)
YELLOW = (255, 205, 40)
GREEN = (60, 200, 110)
RED = (230, 70, 70)
ORANGE = (255, 150, 40)
CYAN = (70, 210, 230)
BLUE = (60, 130, 255)
PURPLE = (120, 60, 200)

AGENT_COLORS = [(255, 90, 90), (90, 220, 120), (90, 150, 255), (255, 210, 60), (230, 100, 230),
                (70, 220, 230), (255, 150, 50), (170, 120, 255)]


class Fonts:
    def __init__(self):
        self.title = pygame.font.Font(None, 52)
        self.big = pygame.font.Font(None, 36)
        self.normal = pygame.font.Font(None, 28)
        self.small = pygame.font.Font(None, 23)
        self.tiny = pygame.font.Font(None, 19)


def open_window(caption, min_w=1100, min_h=680, max_w=1700, max_h=1000):
    pygame.init()
    info = pygame.display.Info()
    w = max(min_w, min(max_w, (info.current_w - 60) if info.current_w > 0 else 1280))
    h = max(min_h, min(max_h, (info.current_h - 120) if info.current_h > 0 else 720))
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(caption)
    pygame.key.set_repeat(300, 50)
    return screen


def text(surf, txt, font, color, **pos):
    s = font.render(txt, True, color)
    r = s.get_rect(**pos)
    surf.blit(s, r)
    return r


def fit(txt, font, max_w):
    if font.size(txt)[0] <= max_w:
        return txt
    while txt and font.size(txt + "...")[0] > max_w:
        txt = txt[:-1]
    return txt + "..."


def button(surf, font, label, rect, color=PANEL_LIGHT, enabled=True, active=False):
    hovered = enabled and rect.collidepoint(pygame.mouse.get_pos())
    base = color if enabled else (45, 48, 58)
    if hovered:
        base = tuple(min(255, c + 28) for c in base)
    pygame.draw.rect(surf, base, rect, border_radius=8)
    if active:
        pygame.draw.rect(surf, YELLOW, rect, 3, border_radius=8)
    text(surf, fit(label, font, rect.w - 12), font, WHITE if enabled else TEXT_DIM, center=rect.center)


class Stepper:
    """Etiqueta + valor + botones - / +. options: lista de (valor, texto) o rango numérico."""

    def __init__(self, key, label, options=None, lo=None, hi=None, step=None, fmt="{}", help_text=""):
        self.key, self.label = key, label
        self.options = options
        self.lo, self.hi, self.step, self.fmt = lo, hi, step, fmt
        self.help = help_text
        self.rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, x, y, w, h=34):
        self.rect = pygame.Rect(x, y, w, h)
        self.minus = pygame.Rect(x + w - 140, y + 3, 30, h - 6)
        self.plus = pygame.Rect(x + w - 34, y + 3, 30, h - 6)

    def display(self, value):
        if self.options:
            for v, label in self.options:
                if v == value:
                    return label
            return str(value)
        if isinstance(value, bool):
            return "Sí" if value else "No"
        return self.fmt.format(value)

    def change(self, value, direction):
        if isinstance(value, bool):
            return not value
        if self.options:
            values = [v for v, _ in self.options]
            i = values.index(value) if value in values else 0
            return values[max(0, min(len(values) - 1, i + direction))]
        new = value + direction * self.step
        new = max(self.lo, min(self.hi, new))
        return round(new, 3) if isinstance(self.step, float) else int(new)

    def click(self, pos, value):
        if self.minus.collidepoint(pos):
            return self.change(value, -1)
        if self.plus.collidepoint(pos) or (isinstance(value, bool) and self.rect.collidepoint(pos)):
            return self.change(value, +1)
        return None

    def draw(self, surf, fonts, value, highlight=False):
        if highlight:
            pygame.draw.rect(surf, PANEL_LIGHT, self.rect, border_radius=6)
        label_w = (self.rect.right - 80 if isinstance(value, bool) else self.minus.x) - self.rect.x - 14
        text(surf, fit(self.label, fonts.small, label_w), fonts.small, WHITE, midleft=(self.rect.x + 8, self.rect.centery))
        if isinstance(value, bool):
            pill = pygame.Rect(self.rect.right - 70, self.rect.y + 5, 62, self.rect.h - 10)
            pygame.draw.rect(surf, GREEN if value else PANEL_LIGHT, pill, border_radius=12)
            text(surf, "Sí" if value else "No", fonts.small, WHITE, center=pill.center)
            return
        button(surf, fonts.normal, "-", self.minus)
        button(surf, fonts.normal, "+", self.plus)
        val_rect = pygame.Rect(self.minus.right + 4, self.rect.y, self.plus.left - self.minus.right - 8, self.rect.h)
        text(surf, fit(self.display(value), fonts.small, val_rect.w), fonts.small, YELLOW, center=val_rect.center)


class TextInput:
    def __init__(self, value=""):
        self.value = value

    def key(self, event, max_len=24):
        if event.key == pygame.K_BACKSPACE:
            self.value = self.value[:-1]
        elif event.unicode and event.unicode.isprintable() and len(self.value) < max_len:
            self.value += event.unicode


def draw_network(surf, fonts, rect, sizes, activations=None, weights=None, max_nodes=12):
    """Dibuja la red neuronal: nodos por capa y, si hay, sus activaciones"""
    pygame.draw.rect(surf, (28, 30, 40), rect, border_radius=8)
    layers = len(sizes)
    xs = [rect.x + 24 + (rect.w - 48) * i / max(1, layers - 1) for i in range(layers)]
    positions = []
    for li, n in enumerate(sizes):
        shown = min(n, max_nodes)
        ys = [rect.y + 18 + (rect.h - 36) * (j + 0.5) / shown for j in range(shown)]
        positions.append([(xs[li], y) for y in ys])
    for li in range(layers - 1):
        for a in positions[li]:
            for b in positions[li + 1]:
                pygame.draw.line(surf, (60, 64, 80), a, b, 1)
    for li, layer in enumerate(positions):
        acts = activations[li] if activations is not None else None
        best = int(acts.argmax()) if (acts is not None and li == layers - 1) else -1
        for j, (x, y) in enumerate(layer):
            v = float(acts[j]) if acts is not None and j < len(acts) else 0.0
            v = max(-1.0, min(1.0, v))
            color = (int(80 + 175 * max(0, v)), int(80 + 60 * (1 - abs(v))), int(80 + 175 * max(0, -v)))
            pygame.draw.circle(surf, color, (int(x), int(y)), 6)
            if j == best:
                pygame.draw.circle(surf, YELLOW, (int(x), int(y)), 9, 2)
        if sizes[li] > max_nodes:
            text(surf, f"+{sizes[li] - max_nodes}", fonts.tiny, TEXT_DIM, midtop=(layer[-1][0], layer[-1][1] + 8))
