# Sistema de Entrenamiento Genético/Evolutivo

## 🚀 Características Principales

### 1. **Múltiples Agentes Simultáneos**
- Entrena varios agentes al mismo tiempo en la misma pista
- Cada agente tiene su propio color para fácil identificación
- Los agentes compiten entre sí en cada generación

### 2. **Selección Evolutiva**
- Al final de cada generación, se seleccionan los mejores agentes
- Los mejores se clonan y mutan para crear la siguiente generación
- Sistema de elitismo: los mejores agentes pasan directamente a la siguiente generación

### 3. **Menú Interactivo en Tiempo Real**
Ubicado en la esquina superior derecha, permite ajustar:
- **Número de Agentes** (1-8): Cantidad de agentes por generación
- **Max Reward** (50-500): Recompensa máxima objetivo
- **Epsilon** (0.0-1.0): Nivel de exploración aleatoria
- **Velocidad** (0.5-5.0): Multiplicador de velocidad de simulación

### 4. **Visualización Mejorada**
- Cada agente muestra su ID y fitness actual
- Colores distintos para cada agente
- Panel de control con estadísticas en tiempo real

---

## 📋 Cómo Usar

### Ejecutar el Entrenamiento

```bash
python train_genetic.py
```

### Controles

| Tecla | Acción |
|-------|--------|
| **SPACE** | Pausar/Reanudar entrenamiento |
| **ESC** | Salir del entrenamiento |
| **Click Izquierdo** | Ajustar parámetros en el menú |

### Ajustar Parámetros

1. **Número de Agentes**: Click en +/- para cambiar cuántos agentes entrenan simultáneamente
2. **Epsilon**: Controla la exploración (0.0 = sin exploración, 1.0 = máxima exploración)
3. **Velocidad**: Acelera o desacelera la simulación (útil para entrenamientos largos)

---

## 🧬 Cómo Funciona el Algoritmo Genético

### 1. **Inicialización**
- Se crean N agentes con redes neuronales aleatorias
- Cada agente tiene su propio "cerebro" (red neuronal)

### 2. **Evaluación (Fitness)**
- Cada agente conduce hasta chocar o alcanzar el límite de pasos
- El fitness se calcula como: `fitness = recompensa_total + pasos * 0.1`
- Mayor fitness = mejor desempeño

### 3. **Selección**
- Los agentes se ordenan por fitness
- Los mejores 25% pasan directamente (elitismo)
- El resto se elimina

### 4. **Reproducción y Mutación**
- Los mejores agentes se clonan
- Los clones se mutan (cambios aleatorios en los pesos de la red)
- Esto crea variedad genética

### 5. **Nueva Generación**
- Los agentes elite + los mutados forman la nueva generación
- El proceso se repite

---

## 📊 Sistema de Recompensas

### Recompensas Positivas
- **Velocidad**: `reward = velocidad / velocidad_máxima`
- **Supervivencia**: `+0.1` por cada paso sin chocar

### Penalizaciones
- **Colisión**: `-100.0` (termina el episodio)
- **Proximidad a bordes**: `-0.5` si sensor < 20% de distancia máxima

### Fitness Total
```
fitness = recompensa_total + (pasos * 0.1)
```

---

## 💾 Archivos Generados

### `best_genetic_agent.pth`
- Mejor agente de todas las generaciones
- Se guarda automáticamente cuando se supera el récord
- Se carga automáticamente al iniciar

### `genetic_training_log.csv`
Registro de entrenamiento con columnas:
- `Generation`: Número de generación
- `Best_Fitness`: Mejor fitness de la generación
- `Avg_Fitness`: Fitness promedio
- `Num_Agents`: Número de agentes
- `Epsilon`: Valor de epsilon usado

---

## 🎨 Colores de los Agentes

| Color | Agente |
|-------|--------|
| 🔴 Rojo | Agente #0 (usualmente el mejor) |
| 🟢 Verde | Agente #1 |
| 🔵 Azul | Agente #2 |
| 🟡 Amarillo | Agente #3 |
| 🟣 Magenta | Agente #4 |
| 🔷 Cyan | Agente #5 |
| 🟠 Naranja | Agente #6 |
| 🟣 Púrpura | Agente #7 |

---

## ⚙️ Parámetros Recomendados

### Para Entrenamiento Rápido
- **Agentes**: 3-4
- **Epsilon**: 0.2-0.3
- **Velocidad**: 2.0-3.0

### Para Mejor Calidad
- **Agentes**: 6-8
- **Epsilon**: 0.1-0.15
- **Velocidad**: 1.0

### Para Exploración
- **Agentes**: 5
- **Epsilon**: 0.5-0.8
- **Velocidad**: 1.5

---

## 🔧 Diferencias con train_dqn.py

| Característica | train_dqn.py | train_genetic.py |
|----------------|--------------|------------------|
| **Método** | Deep Q-Learning | Algoritmo Genético |
| **Agentes** | 1 a la vez | Múltiples simultáneos |
| **Aprendizaje** | Gradiente descendente | Evolución/Mutación |
| **Memoria** | Experience Replay | No usa memoria |
| **Velocidad** | Más lento | Más rápido |
| **Exploración** | Epsilon-greedy | Mutación genética |
| **Mejor para** | Convergencia precisa | Exploración rápida |

---

## 📈 Consejos para Mejor Entrenamiento

### 1. **Ajusta Epsilon Gradualmente**
- Empieza con epsilon alto (0.3-0.5) para exploración
- Reduce gradualmente a 0.1-0.15 cuando veas progreso

### 2. **Número de Agentes**
- Más agentes = más diversidad pero más lento
- 5-6 agentes es un buen balance

### 3. **Velocidad de Simulación**
- Usa velocidad alta (3.0-5.0) cuando el entrenamiento es estable
- Reduce a 1.0 para observar comportamiento detallado

### 4. **Paciencia**
- Las primeras 10-20 generaciones pueden ser caóticas
- El progreso real suele verse después de 30-50 generaciones

### 5. **Guarda Progreso**
- El mejor agente se guarda automáticamente
- Puedes detener y reanudar el entrenamiento sin perder progreso

---

## 🐛 Solución de Problemas

### Los agentes chocan inmediatamente
- **Solución**: Aumenta epsilon a 0.3-0.5 para más exploración
- Verifica que la posición inicial esté en la pista blanca

### El entrenamiento es muy lento
- **Solución**: Aumenta la velocidad de simulación
- Reduce el número de agentes

### No hay progreso después de muchas generaciones
- **Solución**: Aumenta epsilon temporalmente
- Reinicia con el mejor agente guardado
- Ajusta las recompensas en el código si es necesario

### La ventana se congela
- **Solución**: Los eventos de pygame se procesan correctamente
- Si persiste, reduce la velocidad de simulación

---

## 🎯 Objetivos de Fitness

| Fitness | Nivel |
|---------|-------|
| < 50 | Principiante (choca rápido) |
| 50-200 | Básico (sobrevive un poco) |
| 200-500 | Intermedio (completa parte del circuito) |
| 500-1000 | Avanzado (completa varias vueltas) |
| > 1000 | Experto (conduce establemente) |

---

## 📝 Notas Técnicas

### Arquitectura de Red Neuronal
- **Entrada**: 7 valores (5 sensores + velocidad + ángulo)
- **Capas ocultas**: Definidas en `model.py`
- **Salida**: 4 acciones (nada, acelerar, izq+acel, der+acel)

### Mutación
- **Tasa de mutación**: 20% de los pesos
- **Fuerza de mutación**: ±30% del valor original
- Usa distribución normal para cambios suaves

### Elitismo
- 25% de los mejores agentes pasan sin cambios
- Garantiza que no se pierda el mejor desempeño

---

## 🚀 Próximos Pasos

1. Ejecuta `python train_genetic.py`
2. Observa las primeras generaciones
3. Ajusta parámetros según el comportamiento
4. Deja entrenar por 50-100 generaciones
5. El mejor agente se guarda automáticamente

¡Buena suerte con el entrenamiento! 🏎️💨
