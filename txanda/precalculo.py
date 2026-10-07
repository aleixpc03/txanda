"""Precálculo para la app publicada: los precios de OMIE y las previsiones de las semanas del
backtest, para no descargar ni reentrenar nada en un servidor con poca CPU.

Las previsiones se guardan tal como las hace cada previsor con lo conocido el día de decisión,
así que la app resuelve exactamente lo mismo que el cálculo en vivo; solo se ahorra el tiempo.
Los futuros de OMIP no se guardan: sus condiciones no permiten redistribuirlos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .prevision import HORIZONTES, PrevisorLEAR, Vigilancia

CARPETA = Path(__file__).resolve().parent.parent / "precalculo"
PRECIOS = CARPETA / "precios_omie.csv.gz"
PREVISIONES = CARPETA / "previsiones.npz"
HISTORIA_APP = pd.Timedelta(days=400)  # la que carga app.py antes de cada lunes


class PrevisorGrabado:
    """Devuelve las previsiones guardadas de un previsor; la clave es la fecha del día de decisión."""

    def __init__(self, fechas: pd.DatetimeIndex, tabla: dict):
        self.fechas, self.tabla = fechas, tabla

    def predecir(self, pos_D: int, k: int) -> np.ndarray:
        clave = (self.fechas[pos_D].date(), k)
        if clave not in self.tabla:  # el previsor en vivo tampoco la pudo calcular: falta el día previsto
            raise IndexError(f"sin previsión guardada para {clave}")
        return self.tabla[clave]


def _grabar(previsor, fechas, posiciones) -> dict:
    tabla = {}
    for p in posiciones:
        for k in HORIZONTES:
            try:
                tabla[(fechas[p].date(), k)] = previsor.predecir(p, k)
            except Exception:  # sin dato: al reproducir, la vigilancia verá el mismo fallo
                pass
    return tabla


def grabar_semana(lunes, M, fechas) -> dict:
    """Las previsiones que pide `planificar_semana` sin futuros: LEAR y aprendizaje automático
    desde el domingo hasta el sábado, y el LEAR de la semana anterior para la vigilancia."""
    from .backtest import previsores_semana

    pos = fechas.get_loc(pd.Timestamp(lunes))
    previsores = previsores_semana(lunes, M, fechas)
    anterior = PrevisorLEAR(M, fechas).entrenar(pos - 7)
    semana = range(pos - 1, pos + 6)
    return {"L": _grabar(previsores["L"], fechas, semana), "D": _grabar(previsores["D"], fechas, semana),
            "L_anterior": _grabar(anterior, fechas, range(pos - 8, pos - 1))}


def grabar_semana_app(lunes) -> dict:
    """`grabar_semana` con la misma matriz que carga la app para esa semana."""
    from .precios import matriz_diaria

    matriz = matriz_diaria(lunes - HISTORIA_APP, lunes + pd.Timedelta(days=6))
    return grabar_semana(lunes, matriz.to_numpy(), pd.DatetimeIndex(matriz.index))


def guardar_precios(desde, hasta, ruta: Path = PRECIOS) -> Path:
    from .precios import dias, precios_omie

    serie = pd.concat([precios_omie(f) for f in dias(desde, hasta)])
    ruta.parent.mkdir(exist_ok=True)
    serie.tz_convert("UTC").round(4).to_csv(ruta)
    return ruta


def guardar_previsiones(grabadas: dict, ruta: Path = PREVISIONES) -> Path:
    """`grabadas` = {lunes: grabar_semana(...)}. Un array por previsor, fecha de decisión y horizonte."""
    arrays = {}
    for lunes, semana in grabadas.items():
        for clave, tabla in semana.items():
            for (fecha, k), valores in tabla.items():
                arrays[f"{pd.Timestamp(lunes):%Y%m%d}|{clave}|{fecha:%Y%m%d}|{k}"] = valores
    ruta.parent.mkdir(exist_ok=True)
    np.savez_compressed(ruta, **arrays)
    return ruta


def cargar_precios(ruta: Path = PRECIOS) -> pd.Series | None:
    if not ruta.exists():
        return None
    serie = pd.read_csv(ruta, index_col=0).iloc[:, 0]
    serie.index = pd.to_datetime(serie.index, utc=True)
    return serie


def cargar_previsiones(lunes, ruta: Path = PREVISIONES) -> dict | None:
    """{clave: {(fecha, k): previsión}} de la semana del `lunes`, o None si no está precalculada."""
    if not ruta.exists():
        return None
    prefijo = f"{pd.Timestamp(lunes):%Y%m%d}|"
    semana: dict = {}
    with np.load(ruta) as datos:
        for nombre in datos.files:
            if nombre.startswith(prefijo):
                _, clave, fecha, k = nombre.split("|")
                semana.setdefault(clave, {})[(pd.Timestamp(fecha).date(), int(k))] = datos[nombre]
    return semana or None


def previsores_grabados(lunes, M, fechas, semana: dict) -> tuple[dict, Vigilancia]:
    """Los previsores de la semana y la vigilancia de LEAR a partir de lo guardado."""
    pos = fechas.get_loc(pd.Timestamp(lunes))
    previsores = {"L": PrevisorGrabado(fechas, semana["L"]), "D": PrevisorGrabado(fechas, semana["D"])}
    anterior = PrevisorGrabado(fechas, semana["L_anterior"])
    return previsores, Vigilancia(M, [(pos - 8, anterior), (pos - 1, previsores["L"])])
