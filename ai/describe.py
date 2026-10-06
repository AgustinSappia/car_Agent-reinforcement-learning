"""
Explicación en palabras de un agente: qué significa cada neurona de entrada,
qué hace cada salida y cómo se calcula su puntaje. La usan la pantalla de
detalle del agente y el taller.
"""

import math

import numpy as np

from ai.config import ACTION_SETS, BRAIN_SIZES, KIND_LABELS


def _side(deg):
    if abs(deg) < 1:
        return "derecho al frente"
    if abs(abs(deg) - 180) < 1:
        return "hacia atrás"
    return f"{abs(deg):.0f}° a la {'izquierda' if deg < 0 else 'derecha'}"


def _rel_inputs(what, scale, has_dist=True):
    """Tres entradas (o dos) que dicen dónde está algo respecto del frente del auto"""
    out = [
        (f"{what}: ¿izq. o der.?", f"Seno del ángulo hacia {what.lower()}. Negativo: está a la izquierda; "
                                   f"positivo: a la derecha; cero: justo adelante o atrás."),
        (f"{what}: ¿adelante?", f"Coseno del ángulo hacia {what.lower()}. 1: adelante; -1: atrás."),
    ]
    if has_dist:
        out.append((f"{what}: distancia", f"Qué tan lejos está {what.lower()} (1 = {scale} px, máximo 1,5)."))
    return out


def inputs(cfg, kind=None):
    """[(nombre corto, explicación)] de cada neurona de entrada, en el orden en que entran a la red"""
    kind = kind or cfg.kind
    out = []
    for i, a in enumerate(cfg.sensor_angles()):
        deg = math.degrees(a)
        out.append((f"Sensor {i + 1} ({deg:+.0f}°)",
                    f"Rayo que mira {_side(deg)}. Vale 1 si no ve pared en {cfg.sensor_range} px y baja a 0 "
                    f"cuando la pared está pegada. Es como los ojos del auto."))
    if cfg.use_speed:
        out.append(("Velocidad", f"Su propia velocidad: 0 quieto, 1 a la máxima ({cfg.max_speed:g}). "
                                 "Sirve para saber cuándo frenar antes de una curva."))
    if kind == 'futbol':
        out += _rel_inputs("Pelota", 1000)
        out += [("Pelota: se acerca/aleja", "Velocidad de la pelota hacia adelante del auto (positivo: se aleja "
                                            "hacia donde mira; negativo: viene hacia él)."),
                ("Pelota: se cruza", "Velocidad de la pelota hacia los costados del auto.")]
        out += _rel_inputs("Arco rival", 1500)
        out += _rel_inputs("Arco propio", 1500, has_dist=False)
        if cfg.see_mates:
            out += [(n, d + " Si juega solo, vale 0 (y la distancia 1,5).")
                    for n, d in _rel_inputs("Compañero", 1000)]
        if cfg.see_rivals:
            out += [(n, d + " Si no hay rival, vale 0 (y la distancia 1,5).") for n, d in _rel_inputs("Rival", 1000)]
        if cfg.use_role:
            out.append(("Número de jugador", "0 para el primer jugador del equipo y 1 para el último. Como todos "
                                             "usan el mismo cerebro, esto les permite jugar en puestos distintos "
                                             "(uno ataca, otro defiende)."))
        out.append(("Patada lista", "1 si ya puede patear de nuevo, 0 si la patada se está recargando."))
    elif cfg.use_compass:
        goal = "la salida" if kind == 'laberinto' else "el próximo checkpoint"
        out += [("Brújula: ¿izq. o der.?", f"Seno del ángulo hacia {goal} en línea recta. Negativo: queda a la "
                                           "izquierda; positivo: a la derecha."),
                ("Brújula: ¿adelante?", f"Coseno del ángulo hacia {goal}. 1: adelante; -1: atrás. "
                                        "Ojo: la brújula no sabe de paredes, solo apunta.")]
    return out


def action_name(a):
    steer, gas, brake, kick = a
    if kick:
        return "Patear", ("Acelera y, si tiene la pelota (o la tiene cerca y adelante) y la patada está lista, "
                          "le pega fuerte hacia adelante.")
    turn = {-1: "a la izquierda", 1: "a la derecha"}.get(int(steer))
    if brake:
        return "Frenar", "Frena: baja la velocidad rápido."
    if gas and turn:
        return f"Acelerar {turn[2:]}", f"Acelera mientras dobla {turn}."
    if gas:
        return "Acelerar recto", "Acelera sin doblar."
    if turn:
        return f"Girar {turn[2:]} (sin acelerar)", f"Dobla {turn} dejando que el auto pierda velocidad."
    return "No hacer nada", "Suelta todo: el auto sigue por inercia y va frenando solo."


def outputs(cfg, kind=None):
    """[(nombre, explicación)] de cada neurona de salida"""
    return [action_name(a) for a in cfg.actions(kind)]


def fitness_lines(cfg, kind=None):
    """[(concepto, puntos)] de cómo se calcula el puntaje"""
    kind = kind or cfg.kind
    if kind == 'futbol':
        return [
            ("Gol a favor", f"+{cfg.r_goal}"),
            ("Gol en contra", f"-{cfg.p_conceded}"),
            ("Gol en propio arco (extra)", f"-{cfg.p_own_goal}"),
            ("Toque a la pelota (máx. 30)", f"+{cfg.r_touch}"),
            ("Patada hacia el arco (máx. 15)", f"+{cfg.r_kick}"),
            ("Acercar la pelota al arco (px)", f"+{cfg.r_advance:g}"),
            ("Cerca de la pelota (siempre)", f"hasta +{cfg.r_near}"),
            ("Amontonarse (siempre)", f"hasta -{cfg.p_crowd}"),
            ("Quedarse quieto (siempre)", f"hasta -{cfg.p_idle}"),
        ]
    if kind == 'laberinto':
        out = [("Cuánto se acercó a la salida (por el camino)", "+1 por píxel")]
    else:
        out = [("Avance por la pista, contando vueltas", "+1 por píxel")]
    out.append(("Por cada paso que le sobró al llegar", f"+{cfg.r_fast:g}"))
    if cfg.p_crash:
        out.append(("Cada choque" if cfg.bounce else "Chocar", f"-{cfg.p_crash}"))
    return out


def summary(cfg, kind=None):
    """[(etiqueta, valor)] del resto de los rasgos"""
    kind = kind or cfg.kind
    sizes = cfg.layer_sizes(kind)
    rows = [
        ("Tipo", KIND_LABELS.get(kind, kind)),
        ("Red", f"{' - '.join(map(str, sizes))} ({cfg.num_params(kind)} pesos)"),
        ("Capas ocultas", f"{cfg.brain_size}: {' y '.join(map(str, BRAIN_SIZES[cfg.brain_size]))} neuronas"),
        ("Acciones", ACTION_SETS[cfg.action_set]['label'].split(':')[0] + (" + patear" if kind == 'futbol' else "")),
        ("Vel. máxima / giro", f"{cfg.max_speed:g} / {cfg.turn_speed:.2f} rad"),
        ("Autos por generación", str(cfg.population)),
        ("Élite", f"{cfg.elite_pct} % pasan sin cambios"),
        ("Mutación", f"prob. {cfg.mutation_rate:.2f}, fuerza {cfg.mutation_strength:.2f}"),
        ("Cruza entre padres", "Sí" if cfg.crossover else "No"),
    ]
    if kind == 'futbol':
        rows.append(("Partidos por cerebro", str(cfg.eval_matches)))
    else:
        rows.append(("Pasos / paciencia", f"{cfg.max_steps} / {cfg.patience}"))
        rows.append(("Al chocar", "Rebota y sigue" if cfg.bounce else "Queda afuera"))
    return rows


def importance(brain):
    """Qué tanto usa la red cada entrada: suma de |pesos| que salen de ella, normalizada a 0..1"""
    if not brain:
        return None
    w = np.abs(np.asarray(brain['W'][0]))
    if w.ndim == 3:
        w = w[0]
    imp = w.sum(axis=1)
    top = float(imp.max()) or 1.0
    return imp / top
