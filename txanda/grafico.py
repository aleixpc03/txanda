"""Gráfico de un día: precio y potencia del horno con cada estrategia."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .estrategias import NOMBRES, marcar_caros
from .milp import potencia
from .planta import Planta


def _tramos(mascara: np.ndarray):
    """Pares (inicio, fin) de los tramos consecutivos a True."""
    bordes = np.diff(np.concatenate([[0], mascara.astype(int), [0]]))
    return zip(np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1))


def dibujar_dia(planta: Planta, precios: pd.Series, planes: dict, tabla: pd.DataFrame, ruta: Path, fraccion_cara: float = 0.25) -> Path:
    p = precios.to_numpy(dtype=float)
    T = len(p)
    x = np.arange(T + 1)
    caros = marcar_caros(p, fraccion_cara)

    fig, ejes = plt.subplots(
        1 + len(planes), 1, figsize=(11, 2.2 + 1.3 * len(planes)), sharex=True,
        gridspec_kw={"height_ratios": [2.2] + [1] * len(planes)},
    )
    ax = ejes[0]
    for a, b in _tramos(caros):
        ax.axvspan(a, b, color="#E8B04B", alpha=0.25, lw=0)
    ax.stairs(p, x, color="#2A48C9", lw=1.6)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_ylabel("€/MWh")
    ax.set_title(f"{precios.index[0]:%d-%m-%Y} · precio del mercado diario (sombreado: {fraccion_cara:.0%} de cuartos más caros)", loc="left", fontsize=10)

    pmax = max(planta.perfil_mw) * 1.15
    for ax, (clave, plan) in zip(ejes[1:], planes.items()):
        ax.stairs(potencia(planta, plan.inicios, T), x, fill=True, color="#5B6273", alpha=0.85)
        ax.set_ylim(0, pmax)
        ax.set_ylabel(clave, rotation=0, labelpad=14, fontsize=12, va="center")
        fila = tabla.loc[clave]
        ax.set_title(f"{NOMBRES[clave]} · {fila.coste_total_eur:,.0f} € · {fila.arranques} secuencias".replace(",", "."),
                     loc="right", fontsize=9, pad=3)
    marcas = np.arange(0, T, 8)
    ejes[-1].set_xticks(marcas, [precios.index[i].strftime("%H:%M") for i in marcas])
    ejes[-1].set_xlim(0, T)
    ejes[-1].set_xlabel("hora local · potencia del horno en MW")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def dibujar_semana(planta: Planta, semana, planes: dict, tabla: pd.DataFrame, ruta: Path, previsiones: dict | None = None) -> Path:
    """Precio real (y previsiones hechas el domingo) y potencia del horno con cada estrategia."""
    from .semana import NOMBRES_SEMANA

    p, lim = semana.precios, semana.limites
    T = len(p)
    x = np.arange(T + 1)
    fig, ejes = plt.subplots(
        1 + len(planes), 1, figsize=(14, 2.6 + 1.05 * len(planes)), sharex=True,
        gridspec_kw={"height_ratios": [2.6] + [1] * len(planes)},
    )
    ax = ejes[0]
    ax.stairs(p, x, color="#2A48C9", lw=1.3, label="precio real")
    estilos = {"ingenua": ("#8C5E00", ":"), "futuros": ("#6B3FA0", "-"), "LEAR": ("#A8281C", "-."), "ML": ("#1D7349", "--")}
    for nombre, serie in (previsiones or {}).items():
        color, linea = estilos.get(nombre, ("#586072", "--"))
        ax.stairs(serie, x, color=color, lw=1.1, ls=linea, label=f"previsión {nombre} (domingo)")
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_ylabel("€/MWh")
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    ax.set_title(f"Semana del {semana.lunes:%d-%m-%Y} · precio del mercado diario", loc="left", fontsize=10)
    pmax = max(planta.perfil_mw) * 1.15
    for ax, (clave, plan) in zip(ejes[1:], planes.items()):
        ax.stairs(potencia(planta, plan.inicios, T), x, fill=True, color="#5B6273", alpha=0.85)
        ax.set_ylim(0, pmax)
        ax.set_ylabel(clave, rotation=0, labelpad=16, fontsize=11, va="center")
        fila = tabla.loc[clave]
        ax.set_title(f"{NOMBRES_SEMANA[clave]} · {fila.coste_total_eur:,.0f} € · captura {fila.captura_pct:.0f} %".replace(",", "."),
                     loc="right", fontsize=9, pad=3)
    for ax in ejes:
        for borde in lim[1:-1]:
            ax.axvline(borde, color="#BBBBBB", lw=0.6)
    dias_es = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
    ejes[-1].set_xticks([(lim[i] + lim[i + 1]) / 2 for i in range(7)], [f"{dias_es[i]} {d.index[0]:%d}" for i, d in enumerate(semana.dias)])
    ejes[-1].tick_params(axis="x", length=0)
    ejes[-1].set_xlim(0, T)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


# Paleta categórica de referencia (orden fijo, validada para series adyacentes en líneas).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TINTA, TINTA_SUAVE, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def dibujar_sensibilidad(valores: pd.DataFrame, ruta: Path, titulo: str, etiquetas_x=None,
                         eje_x: str = "coste de arranque de una secuencia",
                         eje_y: str = "% del ahorro máximo capturado\n(0 = horario fijo A, 100 = oráculo O)",
                         techo: pd.Series | None = None) -> Path:
    """Una línea por estrategia a lo largo de los niveles del parámetro (filas de `valores`).

    Los colores siguen a la estrategia, no a su posición: B, OD, C, F, L y D usan siempre
    los mismos. `techo` dibuja el oráculo como referencia en gris discontinuo.
    """
    from .semana import NOMBRES_SEMANA

    colores = dict(zip(["B", "OD", "C", "F", "L", "D"], SERIES))
    x = np.arange(len(valores))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(REJILLA)
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=TINTA_SUAVE, lw=1)
    etiquetas = []
    if techo is not None:
        ax.plot(x, techo.to_numpy(), color=TINTA_SUAVE, lw=1.5, ls="--", label=f"O · {NOMBRES_SEMANA['O']} (techo)")
        etiquetas.append([techo.iloc[-1], "O"])
    for clave in valores.columns:
        y = valores[clave].to_numpy()
        ax.plot(x, y, color=colores[clave], lw=2, marker="o", ms=6, mec="white", mew=1.5, label=f"{clave} · {NOMBRES_SEMANA[clave]}")
        etiquetas.append([y[-1], clave])
    # Etiquetas directas al final de cada línea, separadas para que no se pisen.
    etiquetas.sort()
    separacion = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.045
    for i in range(1, len(etiquetas)):
        etiquetas[i][0] = max(etiquetas[i][0], etiquetas[i - 1][0] + separacion)
    for y, clave in etiquetas:
        ax.annotate(clave, (x[-1], y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=10, color=TINTA)
    if etiquetas_x is None:
        etiquetas_x = [f"{c:,.0f} €".replace(",", ".") for c in valores.index]
    ax.set_xticks(x, etiquetas_x)
    ax.set_xlim(-0.2, x[-1] + 0.45)
    ax.set_xlabel(eje_x, color=TINTA_SUAVE)
    ax.set_ylabel(eje_y, color=TINTA_SUAVE)
    ax.tick_params(colors=TINTA_SUAVE)
    ax.set_title(titulo, loc="left", fontsize=11, color=TINTA)
    ax.legend(loc="upper left", bbox_to_anchor=(1.06, 1), frameon=False, fontsize=9, labelcolor=TINTA)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor="white")
    plt.close(fig)
    return ruta
