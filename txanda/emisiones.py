"""Intensidad de emisiones del sistema eléctrico peninsular (tCO₂-eq/MWh) por cuarto de hora.

Dos fuentes públicas de Red Eléctrica:
- el seguimiento de la demanda (demanda.ree.es): generación por tecnología cada 5 minutos;
- REData: emisiones oficiales diarias por tecnología, con los factores de la metodología
  de REE (carbón 0,95; ciclo combinado 0,37; cogeneración 0,38; residuos 0,24 tCO₂-eq/MWh).

Las categorías de las dos fuentes no casan una a una (cogeneración y residuos van juntos en
la primera), así que el total oficial de cada día se reparte entre sus cuartos de hora en
proporción a lo que generó esa tecnología en cada uno. El total diario coincide siempre con
el de REE. La intensidad es emisiones / demanda peninsular: el factor medio del consumo.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .precios import CLAVES, a_cuartos

URL_GENERACION = ("https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/demandaGeneracionPeninsula"
                  "?callback=x&curva=DEMANDAAU&fecha={fecha:%Y-%m-%d}")
URL_EMISIONES = ("https://apidatos.ree.es/es/datos/generacion/no-renovables-detalle-emisiones-CO2"
                 "?start_date={inicio:%Y-%m-%d}T00:00&end_date={fin:%Y-%m-%d}T23:59&time_trunc=day"
                 "&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741")
CACHE = Path(__file__).resolve().parent.parent / "datos" / "ree"
CAMPOS = ["dem", "car", "cc", "gf", "cogenResto", "vap"]
# Serie oficial de emisiones → campo de generación del que toma la forma horaria
GRUPOS = {
    "Carbón": "car", "Ciclo combinado": "cc", "Turbina de vapor": "vap",
    "Cogeneración": "cogenResto", "Residuos no renovables": "cogenResto",
    "Fuel + Gas": "gf", "Fuel": "gf", "Motores diésel": "gf", "Turbina de gas": "gf",
}


def leer_generacion(texto: str, fecha) -> pd.DataFrame:
    """Generación media (MW) de cada cuarto de hora del día, por hora de reloj."""
    datos = json.loads(texto[texto.index("(") + 1 : texto.rindex(")")])
    df = pd.DataFrame(datos["valoresHorariosGeneracion"])
    df = df[df.ts.str.startswith(f"{pd.Timestamp(fecha):%Y-%m-%d}")].copy()
    # El día del cambio de hora de octubre repite las 2:00 como «2A» y «2B».
    hora = df.ts.str[11:].str.replace(r"^2[AB]", "02", regex=True)
    minuto = hora.str[3:5].astype(int) // 15 * 15
    df["clave"] = hora.str[:2] + ":" + minuto.map("{:02d}".format)
    for campo in CAMPOS:
        if campo not in df:
            df[campo] = 0.0
    return df.groupby("clave")[CAMPOS].mean().reindex(CLAVES).interpolate(limit_direction="both")


def generacion_dia(fecha, cache: Path = CACHE) -> pd.DataFrame:
    fecha = pd.Timestamp(fecha)
    ruta = cache / f"generacion_{fecha:%Y%m%d}.csv"
    if ruta.exists():
        return pd.read_csv(ruta, index_col="clave")
    respuesta = requests.get(URL_GENERACION.format(fecha=fecha), timeout=60)
    respuesta.raise_for_status()
    tabla = leer_generacion(respuesta.text, fecha)
    cache.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index_label="clave")
    return tabla


def emisiones_mes(mes: pd.Period, cache: Path = CACHE) -> pd.DataFrame:
    """Emisiones oficiales diarias (tCO₂-eq) por tecnología de un mes, peninsular."""
    ruta = cache / f"emisiones_{mes}.csv"
    if ruta.exists():
        return pd.read_csv(ruta, index_col="fecha")
    respuesta = requests.get(URL_EMISIONES.format(inicio=mes.start_time, fin=mes.end_time), timeout=60,
                             headers={"Accept": "application/json"})
    respuesta.raise_for_status()
    filas = {}
    for serie in respuesta.json().get("included", []):
        titulo = serie["attributes"]["title"]
        if titulo in GRUPOS:
            for valor in serie["attributes"]["values"]:
                filas.setdefault(valor["datetime"][:10], {})[titulo] = valor["value"]
    tabla = pd.DataFrame.from_dict(filas, orient="index").fillna(0.0).rename_axis("fecha").sort_index()
    cache.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta)
    return tabla


def intensidad_vector(generacion: pd.DataFrame, oficiales: pd.Series) -> np.ndarray:
    """tCO₂-eq/MWh de los 96 cuartos de un día a partir de su generación y sus totales oficiales."""
    emisiones = np.zeros(len(generacion))
    for tecnologia, toneladas in oficiales.items():
        campo = GRUPOS.get(tecnologia)
        if campo is None or toneladas <= 0:
            continue
        forma = generacion[campo].clip(lower=0).to_numpy()
        reparto = forma / forma.sum() if forma.sum() > 0 else np.full(len(forma), 1 / len(forma))
        emisiones += toneladas * reparto
    return emisiones / (generacion["dem"].to_numpy() * 0.25)


@lru_cache(maxsize=1)
def _intensidad_incluida() -> pd.Series | None:
    from .precalculo import cargar_intensidad

    return cargar_intensidad()


def intensidad_dia(fecha, cuartos: pd.DatetimeIndex, cache: Path = CACHE) -> np.ndarray:
    """Intensidad de emisiones en los cuartos de hora reales del día.

    Si la generación del día no está descargada, usa antes la intensidad que viene con el
    repositorio (`precalculo/`), calculada igual con los mismos datos de REE.
    """
    fecha = pd.Timestamp(fecha)
    if cache == CACHE and not (cache / f"generacion_{fecha:%Y%m%d}.csv").exists():
        incluida = _intensidad_incluida()
        if incluida is not None:
            dia = incluida[(incluida.index >= cuartos[0]) & (incluida.index <= cuartos[-1])]
            if len(dia) == len(cuartos):
                return dia.to_numpy(dtype=float)
    oficiales = emisiones_mes(fecha.to_period("M"), cache).loc[f"{fecha:%Y-%m-%d}"]
    return a_cuartos(intensidad_vector(generacion_dia(fecha, cache), oficiales), cuartos)


def descargar(desde, hasta, pausa_s: float = 0.5, cache: Path = CACHE) -> list:
    """Descarga lo que falte entre dos fechas. Devuelve los días que no se pudieron obtener."""
    fallos = []
    for mes in pd.period_range(desde, hasta, freq="M"):
        emisiones_mes(mes, cache)
    for fecha in pd.date_range(desde, hasta, freq="D"):
        if (cache / f"generacion_{fecha:%Y%m%d}.csv").exists():
            continue
        try:
            generacion_dia(fecha, cache)
        except Exception as error:  # un día sin datos no para la descarga
            fallos.append((fecha.date(), str(error)))
        time.sleep(pausa_s)
    return fallos
