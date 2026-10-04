"""Futuros diarios de electricidad de OMIP (España, carga base).

Se descargan de la página pública de datos de mercado de OMIP. Sus condiciones permiten
guardar el contenido para uso no comercial; reproducirlo o publicarlo requiere
autorización escrita de OMIP. Por eso los datos se quedan en datos/omip/, fuera del
repositorio, y solo se usan para el backtest.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL_OMIP = "https://www.omip.pt/en/dados-mercado?date={fecha:%Y-%m-%d}&product=EL&zone=ES&instrument=FTB"
CACHE = Path(__file__).resolve().parent.parent / "datos" / "omip"
CONTRATO_DIA = re.compile(r"FTB D ([A-Z][a-z]\d{2}[A-Z][a-z]{2}-\d{2})")


def leer_sesion(html: str) -> pd.DataFrame:
    """Contratos diarios de una sesión: día de entrega y precio de referencia (€/MWh)."""
    filas = []
    for tabla in re.findall(r"<table.*?</table>", html, flags=re.S):
        etiquetas = list(dict.fromkeys(CONTRATO_DIA.findall(tabla)))
        if not etiquetas:
            continue
        datos = pd.read_html(io.StringIO(tabla))[0]
        cabecera = datos.iloc[0].astype(str).tolist()
        columna = cabecera.index("D (€/MWh)")
        contratos = datos[datos.iloc[:, 0].astype(str).str.startswith("ISIN")]
        precios = pd.to_numeric(contratos.iloc[:, columna], errors="coerce").tolist()
        if len(precios) != len(etiquetas):
            raise ValueError(f"{len(etiquetas)} contratos diarios y {len(precios)} precios en la misma tabla")
        for etiqueta, precio in zip(etiquetas, precios):
            filas.append({"entrega": dt.datetime.strptime(etiqueta[2:], "%d%b-%y").date(), "precio": precio})
    return pd.DataFrame(filas, columns=["entrega", "precio"])


def futuros_sesion(fecha, cache: Path = CACHE) -> pd.DataFrame:
    """Contratos diarios cotizados en la sesión `fecha`. Vacío si ese día no hubo sesión."""
    fecha = pd.Timestamp(fecha).date()
    ruta = cache / f"ftb_d_{fecha:%Y%m%d}.csv"
    if ruta.exists():
        return pd.read_csv(ruta, parse_dates=["entrega"]).assign(entrega=lambda d: d.entrega.dt.date)
    # La web a veces devuelve la página sin la tabla: en día laborable se reintenta antes
    # de darlo por día sin sesión (festivo).
    for intento in range(3):
        respuesta = requests.get(URL_OMIP.format(fecha=fecha), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        respuesta.raise_for_status()
        sesion = leer_sesion(respuesta.text)
        if len(sesion) or fecha.weekday() >= 5:
            break
        time.sleep(5 * (intento + 1))
    cache.mkdir(parents=True, exist_ok=True)
    sesion.to_csv(ruta, index=False)
    return sesion


def tabla_futuros(desde, hasta, pausa_s: float = 1.0, cache: Path = CACHE) -> pd.DataFrame:
    """Todas las sesiones entre dos fechas, descargando con pausa lo que falte."""
    partes = []
    for fecha in pd.date_range(desde, hasta, freq="D"):
        descargada = (cache / f"ftb_d_{fecha:%Y%m%d}.csv").exists()
        sesion = futuros_sesion(fecha, cache)
        if not descargada:
            time.sleep(pausa_s)
        if len(sesion):
            partes.append(sesion.assign(sesion=fecha.date()))
    if not partes:
        return pd.DataFrame(columns=["sesion", "entrega", "precio"])
    return pd.concat(partes, ignore_index=True)[["sesion", "entrega", "precio"]]
