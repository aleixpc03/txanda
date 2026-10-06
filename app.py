"""Prototipo de Txanda: planificar una semana de coladas frente al precio cuartohorario.

    .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import replace
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from txanda.omip import tabla_futuros
from txanda.planificador import PREVISIONES, orden_fabricacion, planificar_semana, serie_potencia
from txanda.planta import Planta
from txanda.precios import matriz_diaria
from txanda.semana import NOMBRES_SEMANA, cargar_semana

PRIMER_LUNES, ULTIMO_LUNES = dt.date(2025, 10, 6), dt.date(2026, 9, 21)
HISTORIA = pd.Timedelta(days=400)
TURNOS = {"Tres turnos (24 h)": ((0.0, 24.0),), "Dos turnos (06:00–22:00)": ((6.0, 22.0),)}
COLORES = {"B": "#2a78d6", "OD": "#eb6834", "C": "#1baf7a", "F": "#eda100", "L": "#e87ba4", "D": "#008300"}
GRIS = "#8a8984"
RAIZ = Path(__file__).resolve().parent
SALIDAS, PUBLICOS = RAIZ / "salidas", RAIZ / "resultados"


def omip_disponible() -> bool:
    """Los futuros de OMIP no se usan en el proyecto: sus condiciones no lo permiten. El código se
    conserva, pero solo se activa con TXANDA_OMIP=1."""
    return os.environ.get("TXANDA_OMIP") == "1"


OMIP = omip_disponible()
OPCIONES = [p for p in PREVISIONES if OMIP or PREVISIONES[p] != "F"]
# La memoria deja las emisiones como trabajo pendiente: la app no las muestra hasta que la memoria las incluya.
MOSTRAR_CO2 = False
COLUMNAS_ORDEN = ["empieza", "termina", "coladas", "toneladas", "energía (MWh)", "precio medio (€/MWh)"]

st.set_page_config(page_title="Txanda", page_icon="🔥", layout="wide")


@st.cache_data(show_spinner=False)
def cargar_datos(lunes: dt.date):
    lunes = pd.Timestamp(lunes)
    matriz = matriz_diaria(lunes - HISTORIA, lunes + pd.Timedelta(days=6))
    futuros = tabla_futuros(lunes - pd.Timedelta(days=10), lunes + pd.Timedelta(days=5)) if OMIP else None
    return matriz.to_numpy(), pd.DatetimeIndex(matriz.index), futuros


@st.cache_data(show_spinner=False)
def calcular(lunes: dt.date, prevision: str, coladas: int, rango: tuple[int, int], coste_arranque: float, turnos: str):
    M, fechas, futuros = cargar_datos(lunes)
    planta = replace(Planta(), coladas_dia=coladas, holgura_menos=coladas - rango[0], holgura_mas=rango[1] - coladas,
                     coste_arranque_eur=float(coste_arranque), turnos=TURNOS[turnos])
    return planta, planificar_semana(planta, cargar_semana(lunes), M, fechas, prevision, futuros, con_emisiones=MOSTRAR_CO2)


def euros(v: float) -> str:
    return f"{v:,.0f} €".replace(",", ".")


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Semana y planta")
    with st.form("parametros"):
        elegido = st.date_input("Semana", value=dt.date(2026, 2, 23), min_value=PRIMER_LUNES, max_value=ULTIMO_LUNES + dt.timedelta(days=6),
                                help="Se planifica de lunes a domingo; cualquier día elige su semana.")
        prevision = st.selectbox("Previsión de D+2 a domingo", OPCIONES, index=0,
                                 help="LEAR y aprendizaje automático se entrenan con el año anterior; la ingenua repite la "
                                      "semana anterior." + (" Futuros OMIP: nivel diario del mercado de futuros con la forma "
                                      "de la semana anterior." if OMIP else ""))
        coladas = st.slider("Coladas por día (media semanal)", 10, 20, 18)
        rango = st.slider("Coladas permitidas cada día", 0, 22, (12, 22),
                          help="La semana siempre suma 7 × coladas por día; esto fija cuánto puede variar cada día.")
        coste_arranque = st.number_input("Coste de arrancar una secuencia (€)", 0, 20_000, 2_500, step=500)
        turnos = st.selectbox("Turnos", list(TURNOS))
        st.form_submit_button("Planificar semana", type="primary", width="stretch")
    st.caption("Planta tipo: horno de arco de 100 t por colada, 40 MWh por colada, secuencias de 4 a 10 coladas, "
               "1 h de cambio de artesa. Precios: OMIE y Red Eléctrica." + (" Emisiones: Red Eléctrica." if MOSTRAR_CO2 else "")
               + (" Futuros: OMIP, solo para uso no comercial." if OMIP else ""))

lunes = elegido - dt.timedelta(days=elegido.weekday())

st.title("Txanda")
st.caption("Planificación de coladas frente al precio cuartohorario de la electricidad · Donostia Meeting Minds 2026")

pestana_semana, pestana_anio = st.tabs(["Planificar una semana", "Resultados del año"])

# ---------------------------------------------------------------- una semana
with pestana_semana:
    if not rango[0] <= coladas <= rango[1]:
        st.error(f"El rango diario ({rango[0]}–{rango[1]}) tiene que incluir las {coladas} coladas por día.")
        st.stop()
    try:
        with st.spinner("Planificando la semana día a día…"):
            planta, r = calcular(lunes, prevision, coladas, rango, coste_arranque, turnos)
    except RuntimeError:
        st.error("Con estos turnos no caben las coladas pedidas. Baja las coladas por día o amplía los turnos.")
        st.stop()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    tabla, txanda = r.tabla, r.clave_txanda
    st.markdown(
        f"**Semana del {lunes:%d-%m-%Y}.** Cada día a las 12:00 Txanda conoce los precios de mañana, prevé el resto de la "
        f"semana con *{prevision.lower()}*, fija las coladas de mañana y vuelve a planificar al día siguiente. "
        f"Todas las estrategias producen {7 * coladas} coladas ({7 * coladas * planta.toneladas_colada:,.0f} t).".replace(",", ".")
    )

    fila, base, regla = tabla.loc[txanda], tabla.loc["A"], tabla.loc["B"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Coste con Txanda", euros(fila.coste_total_eur), delta=euros(fila.coste_total_eur - base.coste_total_eur) + " frente al horario fijo",
              delta_color="inverse")
    k2.metric("Ahorro frente al horario fijo", f"{fila.ahorro_vs_A_pct:.1f} %".replace(".", ","))
    k3.metric("Ahorro frente a parar en horas caras", f"{100 * (regla.coste_total_eur - fila.coste_total_eur) / regla.coste_total_eur:.1f} %".replace(".", ","))
    k4.metric("Parte del ahorro máximo capturado", f"{fila.captura_pct:.0f} %", help="0 % = horario fijo, 100 % = oráculo que conoce toda la semana de antemano.")

    # Precio real y previsión del domingo; debajo, potencia del horno con cada plan.
    cuartos = r.semana.cuartos.tz_localize(None)
    precio = pd.concat([
        pd.DataFrame({"hora": cuartos, "€/MWh": r.semana.precios, "serie": "Precio real"}),
        pd.DataFrame({"hora": cuartos, "€/MWh": r.prevision_domingo, "serie": f"Previsión del domingo ({prevision.lower()})"}),
    ])
    x = alt.X("hora:T", title=None, axis=alt.Axis(format="%a %d", labelAngle=0))
    grafico_precio = alt.Chart(precio).mark_line(strokeWidth=1.6, interpolate="step-after").encode(
        x=x, y=alt.Y("€/MWh:Q"),
        color=alt.Color("serie:N", scale=alt.Scale(range=["#2a78d6", GRIS]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(range=[[1, 0], [5, 3]]), legend=None),
        tooltip=[alt.Tooltip("hora:T", format="%a %d %H:%M"), alt.Tooltip("serie:N"), alt.Tooltip("€/MWh:Q", format=".1f")],
    ).properties(height=210)
    claves = [txanda, "A", "O"]
    potencia = serie_potencia(planta, r.semana, r.planes, claves)
    nombres = [f"{c} · {NOMBRES_SEMANA[c]}" for c in claves]
    grafico_potencia = alt.Chart(potencia).mark_area(interpolate="step-after", opacity=0.85).encode(
        x=x, y=alt.Y("MW:Q", title="MW"),
        color=alt.Color("estrategia:N", scale=alt.Scale(domain=nombres, range=[COLORES.get(txanda, "#2a78d6"), GRIS, "#52514e"]), legend=None),
        row=alt.Row("estrategia:N", sort=nombres, title=None, header=alt.Header(labelAngle=0, labelAlign="left", labelAnchor="start")),
        tooltip=[alt.Tooltip("hora:T", format="%a %d %H:%M"), alt.Tooltip("estrategia:N"), alt.Tooltip("MW:Q", format=".0f")],
    ).properties(height=70)
    st.altair_chart(alt.vconcat(grafico_precio, grafico_potencia).resolve_scale(x="shared"), width="stretch")

    st.subheader("Comparación de estrategias")
    columnas = ["estrategia", "coste_total_eur", "eur_por_t", "arranques", "ahorro_vs_A_pct", "captura_pct"]
    columnas += [c for c in ("kgco2_por_t", "emisiones_vs_A_pct") if c in tabla and MOSTRAR_CO2]
    tabla = tabla.assign(estrategia=[f"{e} (Txanda)" if k == txanda else e for k, e in zip(tabla.index, tabla.estrategia)])
    vista = tabla[columnas].rename(columns={
        "estrategia": "Estrategia", "coste_total_eur": "Coste (€)", "eur_por_t": "€/t", "arranques": "Secuencias",
        "ahorro_vs_A_pct": "Ahorro frente a A (%)", "captura_pct": "Ahorro máximo capturado (%)",
        "kgco2_por_t": "kg CO₂/t", "emisiones_vs_A_pct": "Emisiones frente a A (%)"})
    st.dataframe(vista, width="stretch", column_config={
        "Coste (€)": st.column_config.NumberColumn(format="%.0f"), "€/t": st.column_config.NumberColumn(format="%.2f"),
        "Ahorro frente a A (%)": st.column_config.NumberColumn(format="%.1f"),
        "Ahorro máximo capturado (%)": st.column_config.NumberColumn(format="%.0f"),
        "kg CO₂/t": st.column_config.NumberColumn(format="%.2f"),
        "Emisiones frente a A (%)": st.column_config.NumberColumn(format="%+.1f")})
    st.caption(f"Las {len(tabla)} estrategias del benchmark sobre esta semana; Txanda es la que usa la previsión elegida. "
               "A, B y OD producen siempre las mismas coladas cada día; las semanales reparten la semana dentro del rango diario. "
               "El oráculo O conoce todos los precios de antemano: es el techo, no una estrategia posible.")

    st.subheader("Orden de fabricación con Txanda")
    orden = orden_fabricacion(planta, r.semana, r.planes[txanda])[COLUMNAS_ORDEN]
    st.dataframe(orden, width="stretch", hide_index=True, column_config={
        "energía (MWh)": st.column_config.NumberColumn(format="%.0f"),
        "precio medio (€/MWh)": st.column_config.NumberColumn(format="%.1f")})
    st.download_button("Descargar la orden (CSV)", orden.to_csv(index=False).encode("utf-8"),
                       file_name=f"txanda_orden_{lunes:%Y%m%d}.csv", mime="text/csv")

# ---------------------------------------------------------------- resultados del año
with pestana_anio:
    ficheros = [f for f in sorted(SALIDAS.glob("semanas_*_*.csv")) if not f.stem.endswith("_errores")] if OMIP else []
    ficheros = ficheros or sorted(PUBLICOS.glob("semanas.csv"))
    if not ficheros:
        st.info("Aún no hay backtest. Ejecútalo con `python -m txanda semanas 2025-10-06 2026-09-21`.")
    else:
        resultados = pd.read_csv(ficheros[-1], parse_dates=["lunes"])
        # Como la tabla 5 de la memoria: sin el híbrido H, y sin F si no se usan los futuros de OMIP.
        resultados = resultados[~resultados.clave.isin(["H"] if OMIP else ["F", "H"])]
        sumas = ["coste_total_eur", "toneladas"] + (["emisiones_t"] if "emisiones_t" in resultados and MOSTRAR_CO2 else [])
        total = resultados.groupby("clave", sort=False)[sumas].sum()
        a, o = total.loc["A", "coste_total_eur"], total.loc["O", "coste_total_eur"]
        resumen = pd.DataFrame({
            "Estrategia": resultados.groupby("clave", sort=False).estrategia.first(),
            "Coste (M€)": total.coste_total_eur / 1e6,
            "€/t": total.coste_total_eur / total.toneladas,
            "Ahorro frente a A (%)": 100 * (a - total.coste_total_eur) / a,
            "Ahorro máximo capturado (%)": 100 * (a - total.coste_total_eur) / (a - o),
        })
        if "emisiones_t" in total:
            resumen["kg CO₂/t"] = 1000 * total.emisiones_t / total.toneladas
            resumen["Emisiones frente a A (%)"] = 100 * (total.emisiones_t / total.loc["A", "emisiones_t"] - 1)
        st.markdown(f"**Backtest de {resultados.lunes.nunique()} semanas** "
                    f"({resultados.lunes.min():%d-%m-%Y} – {resultados.lunes.max() + pd.Timedelta(days=6):%d-%m-%Y}), "
                    "planta tipo, caso base.")
        st.dataframe(resumen, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.2f" if c in ("Coste (M€)", "€/t") else "%.1f") for c in resumen.columns if c != "Estrategia"})
        for nombre, titulo in [("sensibilidad_arranque", "Sensibilidad al coste de arranque"),
                               ("sensibilidad_flexibilidad", "Sensibilidad a la flexibilidad diaria")]:
            imagenes = (sorted(SALIDAS.glob(f"{nombre}_*.png")) if OMIP else []) or sorted(PUBLICOS.glob(f"{nombre}.png"))
            if imagenes:
                st.subheader(titulo)
                st.image(str(imagenes[-1]), width="stretch")
