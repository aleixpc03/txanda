"""Prototipo de Txanda: planificar una semana de coladas frente al precio cuartohorario.

    .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from txanda.omip import tabla_futuros
from txanda.planificador import PREVISIONES, grafico_semana, nombre_prevision, orden_fabricacion, planificar_semana
from txanda.planta import Planta
from txanda.precalculo import HISTORIA_APP as HISTORIA, cargar_previsiones
from txanda.precios import matriz_diaria
from txanda.semana import cargar_semana

PRIMER_LUNES, ULTIMO_LUNES = dt.date(2025, 10, 6), dt.date(2026, 9, 21)
TURNOS = {"Tres turnos (24 h)": ((0.0, 24.0),), "Dos turnos (06:00–22:00)": ((6.0, 22.0),)}
RAIZ = Path(__file__).resolve().parent
SALIDAS, PUBLICOS = RAIZ / "salidas", RAIZ / "resultados"


def omip_disponible() -> bool:
    """Los futuros de OMIP no se usan en el proyecto: sus condiciones no lo permiten. El código se
    conserva, pero solo se activa con TXANDA_OMIP=1."""
    return os.environ.get("TXANDA_OMIP") == "1"


OMIP = omip_disponible()
OPCIONES = [p for p in PREVISIONES if OMIP or PREVISIONES[p] != "F"]
# Emisiones con la intensidad media de Red Eléctrica (apartado 5.4 de la memoria): quinto indicador
# y columnas de CO₂ en las tablas. En la app publicada salen de precalculo/.
MOSTRAR_CO2 = True
COLUMNAS_ORDEN = ["empieza", "termina", "coladas", "toneladas", "energía (MWh)", "coste energía (€)", "precio medio (€/MWh)"]
DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

st.set_page_config(page_title="Txanda", page_icon="🔥", layout="wide")


@st.cache_data(show_spinner=False)
def cargar_datos(lunes: dt.date):
    lunes = pd.Timestamp(lunes)
    matriz = matriz_diaria(lunes - HISTORIA, lunes + pd.Timedelta(days=6))
    futuros = tabla_futuros(lunes - pd.Timedelta(days=10), lunes + pd.Timedelta(days=5)) if OMIP else None
    return matriz.to_numpy(), pd.DatetimeIndex(matriz.index), futuros


@st.cache_data(show_spinner=False)
def calcular(lunes: dt.date, prevision: str, coladas: int, rango: tuple[int, int], coste_arranque: float, turnos: str,
             con_emisiones: bool, completa: bool = False):
    """Sin `completa`, solo A, B, O y Txanda (indicadores, gráfico y orden); con `completa`, el resto
    de la comparación, reutilizando lo ya resuelto. `con_emisiones` va en los argumentos para que
    forme parte de la clave de la caché."""
    M, fechas, futuros = cargar_datos(lunes)
    planta = replace(Planta(), coladas_dia=coladas, holgura_menos=coladas - rango[0], holgura_mas=rango[1] - coladas,
                     coste_arranque_eur=float(coste_arranque), turnos=TURNOS[turnos])
    previo = calcular(lunes, prevision, coladas, rango, coste_arranque, turnos, con_emisiones)[1] if completa else None
    return planta, planificar_semana(planta, cargar_semana(lunes), M, fechas, prevision, futuros, con_emisiones=con_emisiones,
                                     grabadas=cargar_previsiones(lunes), completa=completa, previo=previo)


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
            planta, r = calcular(lunes, prevision, coladas, rango, coste_arranque, turnos, MOSTRAR_CO2)
    except RuntimeError:
        st.error("Con estos turnos no caben las coladas pedidas. Baja las coladas por día o amplía los turnos.")
        st.stop()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    tabla, txanda = r.tabla, r.clave_txanda
    st.markdown(
        f"**Semana del {lunes:%d-%m-%Y}.** Cada día, tras la subasta de las 12:00, Txanda conoce los precios de mañana, prevé el resto de la "
        f"semana con *{nombre_prevision(prevision)}*, fija las coladas de mañana y vuelve a planificar al día siguiente. "
        f"Todas las estrategias producen {7 * coladas} coladas ({euros(7 * coladas * planta.toneladas_colada)[:-2]} t)."
    )
    if txanda == "LR":
        respaldo = r.planes[txanda].respaldo
        if respaldo:
            st.info("**Vuelta al óptimo diario.** Estos días Txanda no pudo usar la previsión: repartió a partes iguales "
                    "las coladas que quedaban y colocó las de mañana con su precio real.\n\n"
                    + "\n".join(f"- {DIAS_SEMANA[j]}: {motivo}." for j, motivo in respaldo))
        if r.alarmas:
            st.warning("**Aviso al planificador.** El plan no cambia, pero conviene revisarlo con más cuidado:\n\n"
                       + "\n".join(f"- {DIAS_SEMANA[j]}: {aviso}." for j, aviso in r.alarmas))
        if not respaldo and not r.alarmas:
            st.caption("Vigilancia de la previsión: esta semana no han faltado datos y LEAR ha acertado más que la previsión "
                       "ingenua. Si faltan datos, Txanda vuelve al óptimo diario; si LEAR acierta menos que la ingenua, avisa.")

    fila, base, regla = tabla.loc[txanda], tabla.loc["A"], tabla.loc["B"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Coste con Txanda", euros(fila.coste_total_eur), delta=euros(fila.coste_total_eur - base.coste_total_eur) + " frente al horario fijo",
              delta_color="inverse")
    k2.metric("Ahorro frente al horario fijo", f"{fila.ahorro_vs_A_pct:.1f} %".replace(".", ","))
    k3.metric("Ahorro frente a parar en horas caras", f"{100 * (regla.coste_total_eur - fila.coste_total_eur) / regla.coste_total_eur:.1f} %".replace(".", ","))
    k4.metric("Parte del ahorro máximo capturado", f"{fila.captura_pct:.0f} %", help="0 % = horario fijo, 100 % = oráculo que conoce toda la semana de antemano.")
    if MOSTRAR_CO2 and "kgco2_por_t" in tabla:
        k5.metric("Emisiones de CO₂", f"{fila.kgco2_por_t:.1f} kg/t".replace(".", ","), delta_color="inverse",
                  delta=f"{fila.emisiones_vs_A_pct:+.1f} % frente al horario fijo".replace(".", ","),
                  help="Emisiones del consumo del horno con la intensidad media del sistema eléctrico peninsular "
                       "en cada cuarto de hora (Red Eléctrica). Txanda no las minimiza: solo minimiza el coste.")
    elif MOSTRAR_CO2:
        k5.metric("Emisiones de CO₂", "sin datos", help="Faltan los datos de Red Eléctrica de esta semana.")

    # Precio real y previsión del domingo; debajo, potencia del horno con Txanda, el horario fijo y el oráculo.
    st.altair_chart(grafico_semana(planta, r, prevision), width="stretch")

    # La comparación va aquí, pero se calcula después de la orden de fabricación, que ya está lista.
    comparacion = st.container()

    st.subheader("Orden de fabricación con Txanda")
    orden = orden_fabricacion(planta, r.semana, r.planes[txanda])[COLUMNAS_ORDEN]
    st.dataframe(orden, width="stretch", hide_index=True, column_config={
        "energía (MWh)": st.column_config.NumberColumn(format="%.0f"),
        "coste energía (€)": st.column_config.NumberColumn(format="%.0f"),
        "precio medio (€/MWh)": st.column_config.NumberColumn(format="%.1f")})
    st.download_button("Descargar la orden (CSV)", orden.to_csv(index=False).encode("utf-8"),
                       file_name=f"txanda_orden_{lunes:%Y%m%d}.csv", mime="text/csv")

with comparacion:
    st.subheader("Comparación de estrategias")
    with st.spinner("Resolviendo las demás estrategias del benchmark…"):
        tabla = calcular(lunes, prevision, coladas, rango, coste_arranque, turnos, MOSTRAR_CO2, completa=True)[1].tabla
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

# ---------------------------------------------------------------- resultados del año
def resumen_anual(resultados: pd.DataFrame, k_eur: bool = False) -> pd.DataFrame:
    """Tabla del backtest como la tabla 4 de la memoria; con `k_eur`, también el ahorro anual en k€."""
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
    if k_eur:
        resumen.insert(4, "Ahorro frente a A (k€/año)", (a - total.coste_total_eur) * 52 / resultados.lunes.nunique() / 1000)
    if "emisiones_t" in total:
        resumen["kg CO₂/t"] = 1000 * total.emisiones_t / total.toneladas
        resumen["Emisiones frente a A (%)"] = 100 * (total.emisiones_t / total.loc["A", "emisiones_t"] - 1)
    return resumen


def mostrar_tabla(resumen: pd.DataFrame):
    formatos = {"Coste (M€)": "%.2f", "€/t": "%.2f", "Ahorro frente a A (k€/año)": "%.0f"}
    st.dataframe(resumen, width="stretch", column_config={
        c: st.column_config.NumberColumn(format=formatos.get(c, "%.1f")) for c in resumen.columns if c != "Estrategia"})


def describir_escenario(cambios: dict) -> str:
    """Frase con lo que cambia el escenario respecto a la planta tipo."""
    base, partes = Planta(), []
    if "coste_arranque_eur" in cambios:
        partes.append(f"arranques de {euros(cambios['coste_arranque_eur'])} ({euros(base.coste_arranque_eur)} en el caso base)")
    if "secuencia_min" in cambios:
        partes.append(f"secuencias de al menos {cambios['secuencia_min']} coladas ({base.secuencia_min})")
    if "secuencia_max" in cambios:
        partes.append(f"secuencias de como mucho {cambios['secuencia_max']} coladas ({base.secuencia_max})")
    if "holgura_menos" in cambios or "holgura_mas" in cambios:
        minimo = base.coladas_dia - cambios.get("holgura_menos", base.holgura_menos)
        maximo = base.coladas_dia + cambios.get("holgura_mas", base.holgura_mas)
        partes.append(f"entre {minimo} y {maximo} coladas al día ({base.coladas_dia_min}–{base.coladas_dia_max})")
    if "cambio_secuencia_cuartos" in cambios:
        minutos = 15 * cambios["cambio_secuencia_cuartos"]
        partes.append(f"{minutos} minutos de cambio de artesa ({15 * base.cambio_secuencia_cuartos} minutos)")
    return ", ".join(partes)


with pestana_anio:
    ficheros = [f for f in sorted(SALIDAS.glob("semanas_*_*.csv")) if not f.stem.endswith("_errores")] if OMIP else []
    ficheros = ficheros or sorted(PUBLICOS.glob("semanas.csv"))
    if not ficheros:
        st.info("Aún no hay backtest. Ejecútalo con `python -m txanda semanas 2025-10-06 2026-09-21`.")
    else:
        resultados = pd.read_csv(ficheros[-1], parse_dates=["lunes"])
        # Como la tabla 4 de la memoria: sin el híbrido H, y sin F si no se usan los futuros de OMIP.
        resultados = resultados[~resultados.clave.isin(["H"] if OMIP else ["F", "H"])]
        st.markdown(f"**Backtest de {resultados.lunes.nunique()} semanas** "
                    f"({resultados.lunes.min():%d-%m-%Y} – {resultados.lunes.max() + pd.Timedelta(days=6):%d-%m-%Y}), "
                    "planta tipo, caso base.")
        mostrar_tabla(resumen_anual(resultados))

        # Escenarios de planta publicados con `python -m txanda publicar`
        for fichero in sorted((PUBLICOS / "escenarios").glob("*.csv")):
            escenario = pd.read_csv(fichero, parse_dates=["lunes"])
            escenario = escenario[~escenario.clave.isin(["F", "H"])]
            parametros = fichero.with_suffix(".json")
            cambios = json.loads(parametros.read_text())["cambios"] if parametros.exists() else {}
            st.subheader(f"Escenario {fichero.stem.replace('_', ' ')}")
            st.markdown(f"Mismo backtest de {escenario.lunes.nunique()} semanas con otra planta"
                        + (f": {describir_escenario(cambios)}." if cambios else "."))
            mostrar_tabla(resumen_anual(escenario, k_eur=True))
            nota = ("Con otros costes de arranque el coste por tonelada cambia mucho, así que los porcentajes no se pueden "
                    "comparar con el caso base: para comparar escenarios, mejor el ahorro en k€ al año.")
            if "optimo_demostrado" in escenario and (~escenario.optimo_demostrado).any():
                sin_optimo = int((~escenario.optimo_demostrado).sum())
                nota += (f" En {sin_optimo} de {len(escenario)} problemas el solver agotó el tiempo y se usó la mejor solución "
                         "encontrada, así que la parte del ahorro máximo es aproximada.")
            st.caption(nota)

        for nombre, titulo in [("sensibilidad_arranque", "Sensibilidad al coste de arranque"),
                               ("sensibilidad_flexibilidad", "Sensibilidad a la flexibilidad diaria")]:
            imagenes = (sorted(SALIDAS.glob(f"{nombre}_*.png")) if OMIP else []) or sorted(PUBLICOS.glob(f"{nombre}.png"))
            if imagenes:
                st.subheader(titulo)
                st.image(str(imagenes[-1]), width="stretch")
