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
env = Environment(width=WIDTH, height=HEIGHT)

# Reset inicial
observation = env.reset()

# CORREGIDO: Dar velocidad inicial al auto para que se mueva inmediatamente
env.car.speed = 2.0  # Velocidad inicial moderada

# Variables para simulación
running = True
font = pygame.font.Font(None, 24)  # Para mostrar info opcional

print("Controles: ↑ Acelerar, ← Girar Izquierda, → Girar Derecha, ESC para salir")
print("El auto se reinicia automáticamente al chocar.")
print("El auto comienza con velocidad inicial para moverse automáticamente.")

while running:
    # Manejar eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
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
    
    # CORREGIDO: Reiniciar automáticamente al chocar
    if done:
        observation = env.reset()
        env.car.speed = 2.0  # Restaurar velocidad inicial después de reset
    
    # Renderizar el entorno
    env.render(screen)
    
    # Mostrar info en pantalla (para debugging)
    if done:
        text = font.render("¡Choque! Reiniciando...", True, (255, 0, 0))
        screen.blit(text, (10, 10))
    else:
        x, y = info['position']
        min_sensor = info.get('min_sensor', 0)
        text_vel = font.render(f"Vel: {info['speed']:.1f} | Pos: ({x:.0f}, {y:.0f})", True, (255, 255, 255))
        text_sensor = font.render(f"Sensor mín: {min_sensor:.0f} | Recompensa: {reward:.2f}", True, (255, 255, 255))
        
        # Fondo negro para mejor legibilidad
        pygame.draw.rect(screen, (0, 0, 0), (5, 5, 400, 50))
        screen.blit(text_vel, (10, 10))
        screen.blit(text_sensor, (10, 30))
    
    # Actualizar pantalla
    pygame.display.flip()
    
    # Limitar FPS
    clock.tick(60)

# Limpiar y salir
pygame.quit()
sys.exit()
