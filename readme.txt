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