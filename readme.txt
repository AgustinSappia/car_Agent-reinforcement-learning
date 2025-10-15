# car_ai_project/
# 
# Este es el código completo para la base del proyecto "Self Driving Car AI".
# He implementado la estructura solicitada utilizando PyGame para el entorno visual y
# simulación. No se usa PyTorch en esta versión inicial, ya que se enfoca solo en el
# entorno (preparado para integrar un agente DQN más adelante). La pista se genera
# proceduralmente en código (sin necesidad de archivos PNG reales por ahora; puedes
# reemplazar con assets/track.png y assets/car.png si los tienes).
# 
# Para ejecutar: Instala PyGame (`pip install pygame`), luego corre `python main.py`.
# 
# Controles manuales en main.py:
# - Flecha ↑: Acelerar
# - Flecha ←: Girar izquierda
# - Flecha →: Girar derecha
# 
# El entorno está diseñado como un "Gym-like" (reset, step, render), listo para DQN:
# - Observación: Lista de 5 distancias de sensores (floats, normalizables).
# - Acción: Tupla (steering: -1/0/1, throttle: 0/1) – para DQN, puedes discretizar en 5-9 acciones.
# - Recompensa: +1 por paso sin choque, -10 por choque.
# - Done: True en choque (reinicia automáticamente para testing manual).
# 
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