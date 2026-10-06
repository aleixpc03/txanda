"""Resolución de un día a partir de su fecha, para lanzarla en paralelo."""
from __future__ import annotations

import pandas as pd

from .estrategias import comparar_dia
from .planta import Planta
from .precios import a_cuartos, perfil_mes_anterior, precios_omie


def resolver_fecha(fecha, fraccion_cara: float = 0.25) -> pd.DataFrame:
    precios = precios_omie(fecha)
    perfil = a_cuartos(perfil_mes_anterior(fecha), precios.index)
    tabla, _ = comparar_dia(Planta(), precios, perfil, fraccion_cara)
    tabla.insert(0, "fecha", fecha)
    return tabla.reset_index(names="clave")


def previsores_semana(lunes, M, fechas, futuros=None, hibrido: bool = True) -> dict:
    """Previsores de las estrategias semanales, entrenados con lo conocido el domingo."""
    from .prevision import PrevisorFuturos, PrevisorHibrido, PrevisorLEAR, PrevisorML

    pos = fechas.get_loc(pd.Timestamp(lunes))
    previsores = {"F": PrevisorFuturos(M, fechas, futuros)} if futuros is not None and len(futuros) else {}
    previsores["L"] = PrevisorLEAR(M, fechas).entrenar(pos)
    if hibrido and "F" in previsores:
        previsores["H"] = PrevisorHibrido(previsores["L"], previsores["F"], desde_k=3)
    previsores["D"] = PrevisorML(M, fechas).entrenar(pos)
    return previsores


def resolver_semana(lunes, M, fechas, fraccion_cara: float = 0.25, futuros=None, cambios: dict | None = None,
                    gap: float = 0.0):
    """Entrena las previsiones con lo conocido el domingo y simula la semana.

    `cambios` modifica la planta tipo (por ejemplo, {"coste_arranque_eur": 20000}).
    """
    from dataclasses import replace

    from . import semana as modulo_semana
    from .semana import cargar_semana, intensidad_semana, simular_semana

    modulo_semana.GAP = gap
    previsores = previsores_semana(lunes, M, fechas, futuros)
    semana = cargar_semana(lunes)
    planta = replace(Planta(), **(cambios or {}))
    tabla, _, errores = simular_semana(planta, semana, M, fechas, previsores, fraccion_cara=fraccion_cara,
                                       intensidad=intensidad_semana(semana))
    tabla.insert(0, "lunes", lunes)
    errores.insert(0, "lunes", lunes)
    return tabla.reset_index(names="clave"), errores


def resolver_semana_variantes(lunes, M, fechas, variantes: dict, fraccion_cara: float = 0.25, futuros=None):
    """Simula la semana con varias versiones de la planta, {etiqueta: cambios sobre Planta()}.

    Las previsiones se entrenan una vez: no dependen de la planta. Cada variante cambia a
    la vez las decisiones y la evaluación.
    """
    from dataclasses import replace

    from .semana import cargar_semana, simular_semana

    previsores = previsores_semana(lunes, M, fechas, futuros, hibrido=False)
    semana = cargar_semana(lunes)
    tablas = []
    for etiqueta, cambios in variantes.items():
        tabla, _, _ = simular_semana(replace(Planta(), **cambios), semana, M, fechas, previsores, fraccion_cara=fraccion_cara)
        tabla.insert(0, "variante", etiqueta)
        tabla.insert(0, "lunes", lunes)
        tablas.append(tabla.reset_index(names="clave"))
    return pd.concat(tablas, ignore_index=True)


def resolver_semana_sensibilidad(lunes, M, fechas, costes_arranque, fraccion_cara: float = 0.25, futuros=None):
    """Variantes de coste de arranque (se mantiene por compatibilidad con los resultados ya guardados)."""
    variantes = {float(c): {"coste_arranque_eur": float(c)} for c in costes_arranque}
    tabla = resolver_semana_variantes(lunes, M, fechas, variantes, fraccion_cara, futuros)
    return tabla.rename(columns={"variante": "coste_arranque_eur"})
