![image alt](https://github.com/AgustinSappia/car_Agent-reinforcement-learning/blob/ddf8001a2a12138849aa2fee5502b2204fd58ee8/Simulacion.png),
![image alt](https://github.com/AgustinSappia/car_Agent-reinforcement-learning/blob/ddf8001a2a12138849aa2fee5502b2204fd58ee8/Editor%20de%20pistas.png),
![image alt](https://github.com/AgustinSappia/car_Agent-reinforcement-learning/blob/ddf8001a2a12138849aa2fee5502b2204fd58ee8/Selector%20de%20pistas.png),
# car_ai_project/

 Este es el código completo para la base del proyecto "Self Driving Car AI".
 He implementado la estructura solicitada utilizando PyGame para el entorno visual y
 simulación. No se usa PyTorch en esta versión inicial, ya que se enfoca solo en el
 entorno (preparado para integrar un agente DQN más adelante). La pista se genera
 proceduralmente en código (sin necesidad de archivos PNG reales por ahora; puedes
 reemplazar con assets/track.png y assets/car.png si los tienes).

# Para ejecutar: Instala PyGame (`pip install pygame`), luego corre `python main.py`.

# Controles manuales en main.py:
- Flecha ↑: Acelerar
- Flecha ←: Girar izquierda
- Flecha →: Girar derecha
# 
 El entorno está diseñado como un "Gym-like" (reset, step, render), listo para DQN:
 - Observación: Lista de 5 distancias de sensores (floats, normalizables).
 - Acción: Tupla (steering: -1/0/1, throttle: 0/1) – para DQN, puedes discretizar en 5-9 acciones.
 - Recompensa: +1 por paso sin choque, -10 por choque.
 - Done: True en choque (reinicia automáticamente para testing manual).
 
# Código orientado a objetos, limpio y comentado.


. Detección de Agentes Atascados (Nuevo)
Rastrea la posición del agente cada step
Si se mueve menos de 5 píxeles: contador aumenta
50 steps atascado: Penalización de -2.0
150 steps atascado: Muerte automática + penalización de -100
2. Penalización por Estancamiento (Mejorada)
Aumentada de -0.5 a -1.0 por 100 steps sin progreso
Incentiva exploración activa
3. Timeout Global (Nuevo)
2000 steps máximo por agente
Si excede: muerte automática + penalización de -50
Garantiza que la generación siempre termine



Checkpoint: +100 puntos
Meta correcta: +200 puntos
Completar todas las vueltas: +1000 puntos
Cruzar sin checkpoints: -50 puntos
Dirección incorrecta: -100 puntos
Colisión: -200 puntos
Zonas: ±0.3-0.5 (pequeño bonus/malus)

## Entrenamiento genético (nuevo)

`python train.py` (o `python train_genetic2.py`, que es un atajo):

1. Menú principal: **Recorrer una pista** (abre el selector de pistas) o **Resolver un laberinto**.
2. **Taller del agente**: qué ve (sensores, ángulo, alcance, velocidad, brújula), cómo piensa (tamaño de la red),
   cómo maneja (acciones, velocidad, giro) y cómo evoluciona (población, élite, mutación, cruza).
   Se guarda como perfil en `agentes/`.
3. Entrenamiento: velocidades x1 a x40 y Turbo, gráfico de fitness y la red del líder en vivo.

El código está en `ai/`: `config.py` (perfil), `brain.py` (redes de toda la población en numpy),
`world.py` (física y sensores vectorizados), `scenarios.py` (pista y laberinto), `trainer.py`, `menus.py`.
