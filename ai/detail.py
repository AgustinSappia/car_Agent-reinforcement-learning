"""
Pantalla "Detalle del agente": todas sus neuronas de entrada y de salida explicadas,
la red con sus conexiones más fuertes (si ya está entrenado) y cómo se calcula su puntaje.
"""

import numpy as np
import pygame

from ai import ui, describe
from ai.sprites import draw_car


def _wrap(surf, txt, font, color, rect, line_h):
    words, line, y = txt.split(), "", rect.y
    for w in words:
        test = (line + " " + w).strip()
        if font.size(test)[0] > rect.w and line:
            ui.text(surf, line, font, color, topleft=(rect.x, y))
            y += line_h
            line = w
        else:
            line = test
    if line:
        ui.text(surf, line, font, color, topleft=(rect.x, y))
    return y + line_h


class DetailScreen:
    def __init__(self, screen, cfg, kind=None, brain=None, title=None):
        self.screen = screen
        self.fonts = ui.Fonts()
        self.cfg = cfg
        self.kind = kind or cfg.kind
        self.brain = brain
        self.title = title
        self.inputs = describe.inputs(cfg, self.kind)
        self.outputs = describe.outputs(cfg, self.kind)
        self.sizes = cfg.layer_sizes(self.kind)
        self.imp = describe.importance(brain) if brain and brain['sizes'] == self.sizes else None
        self.scroll = 0

    # -------------------------------------------------------------- #
    def layout(self):
        W, H = self.screen.get_size()
        top = 96
        self.in_box = pygame.Rect(24, top, int(W * 0.30), H - top - 140)
        self.out_box = pygame.Rect(W - int(W * 0.30) - 24, top, int(W * 0.30), H - top - 140)
        self.net_box = pygame.Rect(self.in_box.right + 16, top, self.out_box.x - self.in_box.right - 32, H - top - 140)
        self.help_box = pygame.Rect(24, H - 128, W - 260, 104)
        self.back = pygame.Rect(W - 214, H - 74, 190, 50)
        self.row_h = 20
        rows_fit = (self.in_box.h - 34) // self.row_h
        self.scroll = max(0, min(self.scroll, len(self.inputs) - rows_fit))
        self.in_rows = []
        for i in range(self.scroll, min(len(self.inputs), self.scroll + rows_fit)):
            y = self.in_box.y + 30 + (i - self.scroll) * self.row_h
            self.in_rows.append((i, pygame.Rect(self.in_box.x, y, self.in_box.w, self.row_h)))
        oy = self.out_box.y + 30
        self.out_rows = [(i, pygame.Rect(self.out_box.x, oy + i * 22, self.out_box.w, 22)) for i in range(len(self.outputs))]
        self.fit_y = oy + len(self.outputs) * 22 + 16

    # -------------------------------------------------------------- #
    def _node_positions(self):
        r = pygame.Rect(self.net_box.x + 20, self.net_box.y + 30, self.net_box.w - 40, self.net_box.h - 80)
        n_layers = len(self.sizes)
        pos = []
        for li, n in enumerate(self.sizes):
            x = r.x + r.w * li / max(1, n_layers - 1)
            if li == 0:
                # Las entradas se alinean con la lista de la izquierda (las que se ven)
                ys = {i: rect.centery for i, rect in self.in_rows}
                pos.append([(x, ys.get(j)) for j in range(n)])
            elif li == n_layers - 1:
                ys = {i: rect.centery for i, rect in self.out_rows}
                pos.append([(x, ys.get(j)) for j in range(n)])
            else:
                shown = min(n, 24)
                pos.append([(x, r.y + r.h * (j + 0.5) / shown) if j < shown else (x, None) for j in range(n)])
        return pos

    def draw_net(self, hover_in, hover_out):
        scr = self.screen
        pygame.draw.rect(scr, (28, 30, 40), self.net_box, border_radius=10)
        pos = self._node_positions()
        W = None
        if self.imp is not None:
            W = [np.asarray(w[0] if np.ndim(w) == 3 else w) for w in self.brain['W']]
        layer_surf = pygame.Surface(self.net_box.size, pygame.SRCALPHA)
        ox, oy = self.net_box.topleft
        for li in range(len(self.sizes) - 1):
            a_pos, b_pos = pos[li], pos[li + 1]
            if W is not None:
                w = W[li]
                limit = np.quantile(np.abs(w), 0.7)
                scale = float(np.abs(w).max()) or 1.0
            for i, pa in enumerate(a_pos):
                if pa[1] is None:
                    continue
                for j, pb in enumerate(b_pos):
                    if pb[1] is None:
                        continue
                    lit = (li == 0 and i == hover_in) or (li == len(self.sizes) - 2 and j == hover_out)
                    if W is None:
                        color = (90, 95, 120, 140 if lit else 40)
                        width = 1
                    else:
                        v = float(w[i, j])
                        if abs(v) < limit and not lit:
                            continue
                        k = abs(v) / scale
                        base = (80, 220, 120) if v > 0 else (240, 90, 90)
                        alpha = int(60 + 170 * k) if (lit or (hover_in is None and hover_out is None)) else 25
                        color = base + (alpha,)
                        width = 1 + int(2 * k)
                    pygame.draw.line(layer_surf, color, (pa[0] - ox, pa[1] - oy), (pb[0] - ox, pb[1] - oy), width)
        scr.blit(layer_surf, self.net_box.topleft)
        for li, layer in enumerate(pos):
            for j, (x, y) in enumerate(layer):
                if y is None:
                    continue
                hl = (li == 0 and j == hover_in) or (li == len(pos) - 1 and j == hover_out)
                color = ui.CYAN if li == 0 else (ui.ORANGE if li == len(pos) - 1 else (150, 155, 175))
                pygame.draw.circle(scr, ui.YELLOW if hl else color, (int(x), int(y)), 7 if hl else 5)
            if 0 < li < len(pos) - 1:
                ui.text(scr, f"{self.sizes[li]} neuronas", self.fonts.tiny, ui.TEXT_DIM,
                        midtop=(layer[0][0], self.net_box.bottom - 44))
        f = self.fonts.tiny
        ui.text(scr, "Entradas", f, ui.CYAN, topleft=(self.net_box.x + 8, self.net_box.y + 6))
        ui.text(scr, "Capas ocultas", f, ui.TEXT_DIM, midtop=(self.net_box.centerx, self.net_box.y + 6))
        ui.text(scr, "Salidas", f, ui.ORANGE, topright=(self.net_box.right - 8, self.net_box.y + 6))
        if W is not None:
            ui.text(scr, "verde: suma  ·  rojo: resta  ·  solo las conexiones más fuertes", f,
                    ui.TEXT_DIM, midbottom=(self.net_box.centerx, self.net_box.bottom - 6))

    # -------------------------------------------------------------- #
    def draw(self):
        f, scr = self.fonts, self.screen
        W, H = scr.get_size()
        self.layout()
        scr.fill(ui.BG)
        ui.text(scr, "DETALLE DEL AGENTE", f.title, ui.YELLOW, topleft=(24, 18))
        sub = self.title or f"Agente de {describe.KIND_LABELS.get(self.kind, self.kind)}"
        if self.imp is None:
            sub += "  ·  todavía sin entrenar" if not self.brain else "  ·  cerebro con otra forma de red"
        ui.text(scr, ui.fit(sub, f.small, W - 420), f.small, ui.TEXT_DIM, topleft=(26, 62))
        draw_car(scr, (W - 70, 46), 0, 1.6, tuple(self.cfg.color),
                 stripe=(tuple(self.cfg.color) if self.kind != 'futbol' else None))
        mouse = pygame.mouse.get_pos()
        hover_in = next((i for i, r in self.in_rows if r.collidepoint(mouse)), None)
        hover_out = next((i for i, r in self.out_rows if r.collidepoint(mouse)), None)

        # Entradas
        ui.text(scr, f"ENTRADAS ({len(self.inputs)})", f.small, ui.CYAN, topleft=(self.in_box.x, self.in_box.y))
        if self.imp is not None:
            ui.text(scr, "cuánto la usa", f.tiny, ui.TEXT_DIM, topright=(self.in_box.right - 14, self.in_box.y + 4))
        for i, r in self.in_rows:
            if i == hover_in:
                pygame.draw.rect(scr, ui.PANEL_LIGHT, r, border_radius=4)
            ui.text(scr, f"{i + 1:>2}", f.tiny, ui.TEXT_DIM, midleft=(r.x + 4, r.centery))
            bar_w = 70 if self.imp is not None else 0
            ui.text(scr, ui.fit(self.inputs[i][0], f.tiny, r.w - 40 - bar_w), f.tiny, ui.WHITE,
                    midleft=(r.x + 28, r.centery))
            if self.imp is not None:
                bar = pygame.Rect(r.right - bar_w - 14, r.y + 6, bar_w, r.h - 12)
                pygame.draw.rect(scr, (45, 48, 60), bar, border_radius=3)
                pygame.draw.rect(scr, ui.CYAN, (bar.x, bar.y, max(2, int(bar.w * float(self.imp[i]))), bar.h),
                                 border_radius=3)
        hidden = len(self.inputs) - len(self.in_rows)
        if hidden > 0:
            ui.text(scr, f"ruedita: {hidden} más", f.tiny, ui.TEXT_DIM, bottomleft=(self.in_box.x + 28, self.in_box.bottom))

        self.draw_net(hover_in, hover_out)

        # Salidas, puntaje y rasgos
        ui.text(scr, f"SALIDAS ({len(self.outputs)}): gana la más alta", f.small, ui.ORANGE,
                topleft=(self.out_box.x, self.out_box.y))
        for i, r in self.out_rows:
            if i == hover_out:
                pygame.draw.rect(scr, ui.PANEL_LIGHT, r, border_radius=4)
            ui.text(scr, ui.fit(f"{i + 1}. {self.outputs[i][0]}", f.small, r.w - 10), f.small, ui.WHITE,
                    midleft=(r.x + 6, r.centery))
        y = self.fit_y
        ui.text(scr, "SU PUNTAJE", f.small, ui.CYAN, topleft=(self.out_box.x, y))
        y += 24
        for concept, pts in describe.fitness_lines(self.cfg, self.kind):
            ui.text(scr, ui.fit(concept, f.tiny, self.out_box.w - 90), f.tiny, ui.TEXT_DIM, topleft=(self.out_box.x, y))
            ui.text(scr, pts, f.tiny, ui.GREEN if pts.lstrip("hasta ").startswith("+") else ui.RED,
                    topright=(self.out_box.right, y))
            y += 18
        y += 12
        ui.text(scr, "RASGOS", f.small, ui.CYAN, topleft=(self.out_box.x, y))
        y += 24
        for label, value in describe.summary(self.cfg, self.kind):
            if y > self.out_box.bottom - 14:
                break
            ui.text(scr, label, f.tiny, ui.TEXT_DIM, topleft=(self.out_box.x, y))
            ui.text(scr, ui.fit(value, f.tiny, self.out_box.w - 150), f.tiny, ui.WHITE, topleft=(self.out_box.x + 150, y))
            y += 18

        # Ayuda
        pygame.draw.rect(scr, ui.PANEL, self.help_box, border_radius=10)
        inner = self.help_box.inflate(-28, -20)
        if hover_in is not None:
            name, desc = self.inputs[hover_in]
            extra = ""
            if self.imp is not None:
                extra = f"  La red le da una importancia de {self.imp[hover_in] * 100:.0f} % respecto de la entrada que más usa."
            ui.text(scr, f"Entrada {hover_in + 1}: {name}", f.normal, ui.CYAN, topleft=inner.topleft)
            _wrap(scr, desc + extra, f.small, ui.WHITE, pygame.Rect(inner.x, inner.y + 30, inner.w, 70), 21)
        elif hover_out is not None:
            name, desc = self.outputs[hover_out]
            ui.text(scr, f"Salida {hover_out + 1}: {name}", f.normal, ui.ORANGE, topleft=inner.topleft)
            _wrap(scr, desc + " En cada paso el auto hace la acción cuya neurona de salida tiene el valor más alto.",
                  f.small, ui.WHITE, pygame.Rect(inner.x, inner.y + 30, inner.w, 70), 21)
        else:
            _wrap(scr, "Pasá el mouse por una entrada o una salida para ver qué significa y con qué neuronas se "
                       "conecta. Las entradas son lo que el auto percibe en cada momento (todas valen entre -1 y 1,5); "
                       "la red las mezcla en las capas ocultas y elige una de las salidas.",
                  f.small, ui.TEXT_DIM, inner, 21)
        ui.button(scr, f.normal, "Volver (Esc)", self.back)
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_d, pygame.K_RETURN):
                    return 'back'
                if event.type == pygame.MOUSEWHEEL:
                    self.scroll -= event.y * 2
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.back.collidepoint(event.pos):
                    return 'back'
            self.draw()
            clock.tick(60)


def agent_detail(screen, cfg, kind=None, brain=None, title=None):
    """Muestra el detalle. Devuelve 'back' o 'quit'"""
    return DetailScreen(screen, cfg, kind, brain, title).run()
