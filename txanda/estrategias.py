"""Estrategias de un día. Todas usan el mismo MILP y la misma planta; solo cambia la señal de coste.

A · Horario fijo: el mejor horario fijo para el perfil medio de precios del mes
    anterior. No mira los precios del día, así que se repite igual todo el mes.
B · Paradas en horas caras: sabe qué cuartos de hora son caros (el `fraccion_cara`
    más caro del día) pero no cuánto cuesta cada uno; los valora todos a la prima
    media y la compara con el coste de arrancar. Es una versión generosa de la
    práctica actual del sector, lo que hace más exigente la comparación.
O · Óptimo del día: precio exacto de cada cuarto. Los precios de D+1 se publican
    tras la subasta de las 12:00 de D, así que dentro de un día es implementable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .milp import Plan, evaluar, resolver, verificar_plan
from .planta import DT_H, Planta

NOMBRES = {"A": "Horario fijo (mes anterior)", "B": "Paradas en horas caras", "O": "Óptimo del día"}
DESEMPATE_EUR = 1e-3  # € por cuarto de retraso: adelanta coladas en caso de empate


def disponibilidad(planta: Planta, cuartos: pd.DatetimeIndex) -> np.ndarray:
    """True en los cuartos de hora que caen dentro de algún turno con personal."""
    hora = cuartos.hour + cuartos.minute / 60
    libre = np.zeros(len(cuartos), dtype=bool)
    for inicio, fin in planta.turnos:
        libre |= (hora >= inicio) & (hora < fin)
    return libre


def coste_colada(planta: Planta, senal: np.ndarray) -> np.ndarray:
    """Σ_k Δt·p_k·señal[s+k] para cada inicio s; NaN donde la colada no cabe en el día."""
    d, T = planta.duracion, len(senal)
    energia = np.asarray(planta.perfil_mw) * DT_H
    c = np.full(T, np.nan)
    for s in range(T - d + 1):
        c[s] = energia @ senal[s : s + d]
    return c


def marcar_caros(precios: np.ndarray, fraccion: float) -> np.ndarray:
    k = int(round(fraccion * len(precios)))
    caros = np.zeros(len(precios), dtype=bool)
    caros[np.argsort(-precios, kind="stable")[:k]] = True
    return caros


def plan_fijo(planta: Planta, perfil: np.ndarray, disponible: np.ndarray) -> Plan:
    """Óptimo frente al perfil medio del mes anterior, llevado a los cuartos del día."""
    c = coste_colada(planta, perfil) + DESEMPATE_EUR * np.arange(len(perfil))
    return resolver(planta, c, planta.coste_arranque_eur, disponible)


def plan_regla(planta: Planta, precios: np.ndarray, disponible: np.ndarray, fraccion_cara: float = 0.25) -> Plan:
    caros = marcar_caros(precios, fraccion_cara)
    prima = precios[caros].mean() - precios[~caros].mean() if caros.any() and (~caros).any() else 0.0
    c = prima * coste_colada(planta, caros.astype(float)) + DESEMPATE_EUR * np.arange(len(precios))
    return resolver(planta, c, planta.coste_arranque_eur, disponible)


def plan_optimo(planta: Planta, precios: np.ndarray, disponible: np.ndarray) -> Plan:
    return resolver(planta, coste_colada(planta, precios), planta.coste_arranque_eur, disponible)


def comparar_dia(planta: Planta, precios: pd.Series, perfil: np.ndarray, fraccion_cara: float = 0.25):
    """Resuelve A, B y O para un día. `perfil` es el del mes anterior en los cuartos del día.

    Devuelve la tabla de resultados y los planes.
    """
    p = precios.to_numpy(dtype=float)
    disponible = disponibilidad(planta, precios.index)
    planes = {
        "A": plan_fijo(planta, perfil, disponible),
        "B": plan_regla(planta, p, disponible, fraccion_cara),
        "O": plan_optimo(planta, p, disponible),
    }
    filas = {}
    for clave, plan in planes.items():
        verificar_plan(planta, disponible, plan.inicios)
        filas[clave] = {"estrategia": NOMBRES[clave], **evaluar(planta, p, plan.inicios), "segundos": plan.segundos}
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    base = tabla.loc["A", "coste_total_eur"]
    tabla["ahorro_vs_A_eur"] = base - tabla["coste_total_eur"]
    tabla["ahorro_vs_A_pct"] = 100 * tabla["ahorro_vs_A_eur"] / base
    return tabla, planes
