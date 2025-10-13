import pygame
import sys
from environment import Environment

# Configuración inicial de PyGame
pygame.init()
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self Driving Car AI - Entorno de Prueba")
clock = pygame.time.Clock()

# Crear instancia del entorno
env = Environment()

# Reset inicial
observation = env.reset()

# Variables para simulación
running = True
font = pygame.font.Font(None, 24)  # Para mostrar info opcional

print("Controles: ↑ Acelerar, ← Girar Izquierda, → Girar Derecha")
print("El auto se reinicia automáticamente al chocar.")

while running:
    # Manejar eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
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
    observation, reward, done, info = env.step(action)
    
    # Renderizar el entorno
    env.render(screen)
    
    # Mostrar info opcional en pantalla (para debugging)
    if done:
        text = font.render("¡Choque! Reiniciando...", True, (255, 0, 0))
    else:
        x, y = info['position']
        text = font.render(f"Vel: {info['speed']:.1f} | Pos: ({x:.0f}, {y:.0f})", True, (0, 0, 0))
    screen.blit(text, (10, 10))
    
    # Actualizar pantalla
    pygame.display.flip()
    
    # Limitar FPS
    clock.tick(60)

# Limpiar y salir
pygame.quit()
sys.exit()