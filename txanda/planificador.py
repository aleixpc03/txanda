"""Lo que necesita el prototipo: planificar una semana, comparar la previsión elegida con
todas las demás estrategias y traducir el plan a una orden de fabricación legible."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .milp import Plan, potencia, secuencias
from .planta import DT_H, Planta
from .precios import a_cuartos
from .prevision import ingenua
from .semana import NOMBRES_SEMANA, Semana, intensidad_semana, simular_semana

DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

# Previsión elegida → clave de la estrategia semanal que la usa
PREVISIONES = {"LEAR": "L", "Futuros OMIP": "F", "Aprendizaje automático": "D", "Ingenua": "C"}


@dataclass
class ResultadoSemana:
    tabla: pd.DataFrame
    planes: dict[str, Plan]
    semana: Semana
    clave_txanda: str
    prevision_domingo: np.ndarray  # previsión hecha el domingo, en los cuartos reales de la semana


def planificar_semana(planta: Planta, semana: Semana, M, fechas, prevision: str, futuros=None,
                      con_emisiones: bool = False) -> ResultadoSemana:
    """Simula la semana con todas las estrategias disponibles; Txanda es la de la previsión elegida.

    Sin futuros de OMIP son siete estrategias (A, B, OD, C, L, D, O); con ellos, nueve (F y H).
    """
    from .backtest import previsores_semana

    clave = PREVISIONES[prevision]
    if clave == "F" and (futuros is None or not len(futuros)):
        raise ValueError("No hay futuros de OMIP descargados para esta semana")
    pos = fechas.get_loc(pd.Timestamp(semana.lunes))
    previsores = previsores_semana(semana.lunes, M, fechas, futuros)
    intensidad = intensidad_semana(semana) if con_emisiones else None
    tabla, planes, _ = simular_semana(planta, semana, M, fechas, previsores, intensidad=intensidad)

    prever = previsores[clave].predecir if clave in previsores else (lambda p, k: ingenua(M, p, k))
    partes = [semana.dias[0].to_numpy(dtype=float)]
    partes += [a_cuartos(prever(pos - 1, i + 1), semana.dias[i].index) for i in range(1, 7)]
    return ResultadoSemana(tabla, planes, semana, clave, np.concatenate(partes))


def orden_fabricacion(planta: Planta, semana: Semana, plan: Plan) -> pd.DataFrame:
    """Una fila por secuencia de colada continua: cuándo empieza y acaba, coladas, energía y coste."""
    cuartos, precios = semana.cuartos, semana.precios
    energia = potencia(planta, plan.inicios, len(precios)) * DT_H
    filas = []
    for grupo in secuencias(planta, plan.inicios):
        inicio, fin = grupo[0], grupo[-1] + planta.duracion
        tramo = slice(inicio, fin)
        empieza, termina = cuartos[inicio], cuartos[fin - 1] + pd.Timedelta(minutes=15)
        filas.append({
            "empieza": f"{DIAS[empieza.weekday()]} {empieza:%d %H:%M}",
            "termina": f"{DIAS[termina.weekday()]} {termina:%d %H:%M}",
            "coladas": len(grupo),
            "toneladas": len(grupo) * planta.toneladas_colada,
            "energía (MWh)": energia[tramo].sum(),
            "coste energía (€)": energia[tramo] @ precios[tramo],
            "precio medio (€/MWh)": (energia[tramo] @ precios[tramo]) / energia[tramo].sum(),
        })
    return pd.DataFrame(filas)


def serie_potencia(planta: Planta, semana: Semana, planes: dict[str, Plan], claves) -> pd.DataFrame:
    """Potencia del horno en formato largo, para dibujar."""
    T = len(semana.precios)
    return pd.concat(
        [pd.DataFrame({"hora": semana.cuartos.tz_localize(None), "MW": potencia(planta, planes[c].inicios, T),
                       "estrategia": f"{c} · {NOMBRES_SEMANA[c]}"}) for c in claves],
        ignore_index=True,
    )
