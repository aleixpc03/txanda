"""Precio del mercado diario español (OMIE), por cuarto de hora.

La fuente oficial son los ficheros diarios de OMIE. Si un día no está descargado, se toma
antes de Red Eléctrica (REData), que publica la misma serie y sirve un mes entero en una
sola petición: hemos comprobado que coincide cuarto a cuarto, también en días horarios y en
los cambios de hora.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TZ = "Europe/Madrid"
URL_OMIE = (
    "https://www.omie.es/es/file-download?parents=marginalpdbc"
    "&filename=marginalpdbc_{fecha:%Y%m%d}.{version}"
)
CACHE = Path(__file__).resolve().parent.parent / "datos" / "omie"
URL_REE = ("https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
           "?start_date={inicio:%Y-%m-%d}T00:00&end_date={fin:%Y-%m-%d}T23:59&time_trunc=hour")
CACHE_REE = Path(__file__).resolve().parent.parent / "datos" / "ree_precios"


def cuartos_del_dia(fecha) -> pd.DatetimeIndex:
    """Inicios de los cuartos de hora del día en hora local: 96, o 92/100 al cambiar la hora."""
    dia = pd.Timestamp(fecha).normalize()
    inicio = dia.tz_localize(TZ)
    fin = (dia + pd.Timedelta(days=1)).tz_localize(TZ)
    return pd.date_range(inicio, fin, freq="15min", inclusive="left")


def leer_marginalpdbc(texto: str) -> np.ndarray:
    """Precios de España de un fichero marginalpdbc de OMIE, ordenados por periodo.

    Cada línea de datos es: año;mes;día;periodo;precio Portugal;precio España;
    """
    filas = [linea.split(";") for linea in texto.splitlines() if linea[:1].isdigit()]
    filas.sort(key=lambda f: int(f[3]))
    return np.array([float(f[5]) for f in filas])


def fichero_en_cache(fecha, cache: Path = CACHE) -> Path | None:
    """Versión más reciente del fichero del día ya descargada, si la hay."""
    fecha = pd.Timestamp(fecha).date()
    rutas = sorted(cache.glob(f"marginalpdbc_{fecha:%Y%m%d}.*"), key=lambda r: int(r.suffix[1:]))
    return rutas[-1] if rutas else None


def _descargar(fecha: dt.date, cache: Path) -> str:
    """Descarga la última versión publicada. OMIE retira la .1 cuando publica una corrección (.2, .3…)."""
    texto, version_ok = None, None
    for version in range(1, 10):
        respuesta = requests.get(URL_OMIE.format(fecha=fecha, version=version), timeout=30)
        if respuesta.status_code == 404:
            if texto is not None:
                break
            continue
        respuesta.raise_for_status()
        contenido = respuesta.content.decode("latin-1")
        if contenido.startswith("MARGINALPDBC"):
            texto, version_ok = contenido, version
    if texto is None:
        raise ValueError(f"OMIE no tiene precios para el {fecha}; ¿aún no se han publicado?")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"marginalpdbc_{fecha:%Y%m%d}.{version_ok}").write_text(texto, encoding="latin-1")
    return texto


@lru_cache(maxsize=64)
def _mes_ree(mes: str, cache: Path) -> pd.Series:
    """Precio de un mes entero según REE, por cuarto de hora. Solo guarda en disco los meses cerrados."""
    periodo = pd.Period(mes, freq="M")
    ruta = cache / f"precios_{mes}.csv"
    if ruta.exists():
        serie = pd.read_csv(ruta, index_col=0).iloc[:, 0]
        serie.index = pd.to_datetime(serie.index, utc=True).tz_convert(TZ)
        return serie
    respuesta = requests.get(URL_REE.format(inicio=periodo.start_time, fin=periodo.end_time), timeout=60,
                             headers={"Accept": "application/json"})
    respuesta.raise_for_status()
    valores = next(s for s in respuesta.json()["included"] if s["attributes"]["title"] == "Precio mercado spot")
    serie = pd.Series({v["datetime"]: v["value"] for v in valores["attributes"]["values"]}, name="precio_eur_mwh")
    serie.index = pd.to_datetime(serie.index, utc=True).tz_convert(TZ)
    serie = serie.sort_index()
    if periodo.end_time.date() < dt.date.today():
        cache.mkdir(parents=True, exist_ok=True)
        serie.tz_convert("UTC").to_csv(ruta)
    return serie


def _precios_ree_dia(fecha: dt.date, cache: Path) -> pd.Series | None:
    """Precios del día según REE, o None si REE no los tiene completos."""
    try:
        mes = _mes_ree(str(pd.Period(fecha, freq="M")), cache)
    except Exception:
        return None
    cuartos = cuartos_del_dia(fecha)
    dia = mes[(mes.index >= cuartos[0]) & (mes.index <= cuartos[-1])]
    if len(dia) != len(cuartos):
        return None
    return pd.Series(dia.to_numpy(dtype=float), index=cuartos, name="precio_eur_mwh")


def precios_omie(fecha, cache: Path = CACHE, cache_ree: Path | None = CACHE_REE) -> pd.Series:
    """Precio marginal de España (€/MWh) para cada cuarto de hora del día.

    Antes del 1-10-2025 el mercado era horario: cada precio se repite en sus
    cuatro cuartos. Usa el fichero de OMIE si ya está en `cache`; si no, la misma serie
    de REE (un mes por petición, guardado en `cache_ree`); si tampoco, descarga de OMIE.
    Con `cache_ree=None` solo se usa OMIE.
    """
    fecha = pd.Timestamp(fecha).date()
    ruta = fichero_en_cache(fecha, cache)
    if ruta is None and cache_ree is not None:
        serie = _precios_ree_dia(fecha, cache_ree)
        if serie is not None:
            return serie
    texto = ruta.read_text(encoding="latin-1") if ruta else _descargar(fecha, cache)

    valores = leer_marginalpdbc(texto)
    cuartos = cuartos_del_dia(fecha)
    if 4 * len(valores) == len(cuartos):
        valores = np.repeat(valores, 4)
    elif len(valores) != len(cuartos):
        raise ValueError(f"{fecha}: {len(valores)} precios para {len(cuartos)} cuartos de hora")
    return pd.Series(valores, index=cuartos, name="precio_eur_mwh")


def dias(desde, hasta) -> list[dt.date]:
    return [d.date() for d in pd.date_range(desde, hasta, freq="D")]


# Cuartos de hora de un día normal por hora de reloj. Sirven para comparar días de
# 92, 96 y 100 cuartos: en los cambios de hora se promedia o se interpola.
CLAVES = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]


def vector_dia(precios: pd.Series) -> np.ndarray:
    """Los precios de un día como vector de 96 cuartos por hora de reloj."""
    v = precios.groupby(precios.index.strftime("%H:%M")).mean().reindex(CLAVES)
    return v.interpolate(limit_direction="both").to_numpy()


def a_cuartos(vector: np.ndarray, cuartos: pd.DatetimeIndex) -> np.ndarray:
    """Lleva un vector de 96 cuartos por hora de reloj a los cuartos reales de un día."""
    return pd.Series(vector, index=CLAVES).reindex(cuartos.strftime("%H:%M")).to_numpy()


def matriz_diaria(desde, hasta) -> pd.DataFrame:
    """Una fila por día y una columna por cuarto de hora de reloj."""
    filas = {f: vector_dia(precios_omie(f)) for f in dias(desde, hasta)}
    return pd.DataFrame.from_dict(filas, orient="index", columns=CLAVES)


@lru_cache(maxsize=None)
def _perfil_mes(mes: str) -> np.ndarray:
    periodo = pd.Period(mes, freq="M")
    return matriz_diaria(periodo.start_time, periodo.end_time.normalize()).mean().to_numpy()


def perfil_mes_anterior(fecha) -> np.ndarray:
    """Precio medio de cada cuarto de hora (96, hora de reloj) en el mes natural anterior a `fecha`."""
    return _perfil_mes(str(pd.Timestamp(fecha).to_period("M") - 1))
