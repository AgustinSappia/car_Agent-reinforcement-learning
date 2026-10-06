"""
Dibujo de los autitos: carrocería, parabrisas, ruedas y luces.

Cada combinación de color, tamaño y ángulo se dibuja una sola vez y se guarda,
así dibujar cientos de autos por cuadro sigue siendo rápido.
"""

import math
from functools import lru_cache

import pygame

CAR_LEN, CAR_WID = 40, 20   # tamaño en el mapa (píxeles a escala 1)
ANGLE_STEP = 4              # grados entre rotaciones guardadas
BASE = 3                    # el dibujo base se hace 3 veces más grande y se achica (bordes suaves)


def _shade(color, k):
    return tuple(max(0, min(255, int(c * k))) for c in color)


@lru_cache(maxsize=64)
def _base(color, stripe):
    """Auto mirando a la derecha (+x), dibujado en grande"""
    L, W = CAR_LEN * BASE, CAR_WID * BASE
    pad = 3 * BASE
    surf = pygame.Surface((L + 2 * pad, W + 2 * pad), pygame.SRCALPHA)
    ox, oy = pad, pad
    tire = (28, 28, 32)
    # Ruedas (asoman a los costados)
    for fx in (0.16, 0.66):
        for fy in (-0.08, 0.86):
            pygame.draw.rect(surf, tire, (ox + L * fx, oy + W * fy, L * 0.2, W * 0.22), border_radius=BASE * 2)
    # Carrocería
    body = pygame.Rect(ox, oy + W * 0.08, L, W * 0.84)
    pygame.draw.rect(surf, _shade(color, 0.55), body.inflate(BASE * 2, BASE * 2), border_radius=BASE * 7)
    pygame.draw.rect(surf, color, body, border_radius=BASE * 6)
    # Franja del equipo (fútbol) a lo largo del techo
    if stripe:
        pygame.draw.rect(surf, stripe, (ox + L * 0.06, oy + W * 0.4, L * 0.88, W * 0.2))
    # Cabina: parabrisas adelante, luneta atrás, techo en el medio
    glass = (40, 55, 75)
    pygame.draw.polygon(surf, glass, [(ox + L * 0.58, oy + W * 0.22), (ox + L * 0.72, oy + W * 0.28),
                                      (ox + L * 0.72, oy + W * 0.72), (ox + L * 0.58, oy + W * 0.78)])
    pygame.draw.polygon(surf, glass, [(ox + L * 0.2, oy + W * 0.26), (ox + L * 0.3, oy + W * 0.22),
                                      (ox + L * 0.3, oy + W * 0.78), (ox + L * 0.2, oy + W * 0.74)])
    pygame.draw.rect(surf, _shade(color, 1.15), (ox + L * 0.31, oy + W * 0.24, L * 0.26, W * 0.52),
                     border_radius=BASE * 2)
    if stripe:
        pygame.draw.rect(surf, stripe, (ox + L * 0.31, oy + W * 0.4, L * 0.26, W * 0.2))
    # Luces: delanteras amarillas, traseras rojas
    for fy in (0.18, 0.7):
        pygame.draw.rect(surf, (255, 240, 160), (ox + L * 0.93, oy + W * fy, L * 0.06, W * 0.12), border_radius=BASE)
        pygame.draw.rect(surf, (210, 40, 40), (ox + L * 0.01, oy + W * fy, L * 0.05, W * 0.12), border_radius=BASE)
    return surf


@lru_cache(maxsize=4096)
def _rotated(color, stripe, size, angle_idx):
    base = _base(color, stripe)
    w = max(4, int(base.get_width() * size / (CAR_LEN * BASE)))
    h = max(2, int(base.get_height() * size / (CAR_LEN * BASE)))
    scaled = pygame.transform.smoothscale(base, (w, h))
    return pygame.transform.rotate(scaled, -angle_idx * ANGLE_STEP)


def draw_car(surf, pos, angle, scale, color, stripe=None, alpha=None):
    """Dibuja un auto centrado en pos (en pantalla), mirando hacia angle (radianes)"""
    size = max(8, int(round(CAR_LEN * scale)))
    idx = int(round(math.degrees(angle) / ANGLE_STEP)) % (360 // ANGLE_STEP)
    img = _rotated(tuple(color), tuple(stripe) if stripe else None, size, idx)
    if alpha is not None:
        img = img.copy()
        img.set_alpha(alpha)
    surf.blit(img, img.get_rect(center=(int(pos[0]), int(pos[1]))))


def variant(color, i):
    """Pequeña variación de un color para distinguir autos de la misma población"""
    k = 0.75 + 0.5 * ((i * 37) % 11) / 10
    return _shade(color, k)


def faded(color):
    """Versión apagada (para el auto de la generación 1 en el modo Expo)"""
    gray = sum(color) / 3
    return tuple(int(c * 0.45 + gray * 0.35) for c in color)
