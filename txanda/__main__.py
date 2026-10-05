"""Línea de órdenes.

    python -m txanda dia 2026-02-23                     un día: tabla y gráfico
    python -m txanda periodo 2025-10-01 2026-09-29      backtest día a día
    python -m txanda semana 2026-02-23                  una semana (lunes): tabla, errores y gráfico
    python -m txanda semanas 2025-10-06 2026-09-21      backtest de todas las semanas entre dos lunes
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import previsores_semana, resolver_fecha, resolver_semana, resolver_semana_sensibilidad, resolver_semana_variantes
from .estrategias import comparar_dia
from .grafico import dibujar_dia, dibujar_semana, dibujar_sensibilidad
from .planta import Planta
from .precios import a_cuartos, dias, fichero_en_cache, matriz_diaria, perfil_mes_anterior, precios_omie

SALIDAS = Path(__file__).resolve().parent.parent / "salidas"
COLUMNAS = ["estrategia", "coste_total_eur", "coste_energia_eur", "arranques", "precio_medio_eur_mwh", "eur_por_t", "ahorro_vs_A_eur", "ahorro_vs_A_pct"]


def _formato(tabla: pd.DataFrame, columnas=None) -> str:
    tabla = tabla[columnas] if columnas else tabla
    return tabla.to_string(float_format=lambda v: f"{v:,.1f}".replace(",", "_").replace(".", ",").replace("_", "."))


def orden_dia(args):
    planta = Planta()
    precios = precios_omie(args.fecha)
    perfil = a_cuartos(perfil_mes_anterior(args.fecha), precios.index)
    tabla, planes = comparar_dia(planta, precios, perfil, args.fraccion_cara)
    print(_formato(tabla, COLUMNAS))
    o, a = tabla.loc["O", "ahorro_vs_A_eur"], tabla.loc["B", "ahorro_vs_A_eur"]
    if o > 0:
        print(f"\nB captura el {100 * a / o:.0f} % del ahorro que consigue O frente a A.")
    ruta = dibujar_dia(planta, precios, planes, tabla, SALIDAS / f"dia_{pd.Timestamp(args.fecha):%Y%m%d}.png", args.fraccion_cara)
    print(f"Gráfico: {ruta}")


def orden_periodo(args):
    fechas, fallos = [], []
    for fecha in dias(args.desde, args.hasta):  # descarga primero, en serie, para no saturar OMIE
        try:
            if fichero_en_cache(fecha) is None:
                precios_omie(fecha)
                time.sleep(0.2)
            fechas.append(fecha)
        except Exception as error:  # un día sin datos no para el backtest
            fallos.append((fecha, str(error)))
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        filas = list(pool.map(resolver_fecha, fechas, repeat(args.fraccion_cara), chunksize=8))
    resultados = pd.concat(filas, ignore_index=True)
    SALIDAS.mkdir(exist_ok=True)
    ruta = SALIDAS / f"periodo_{pd.Timestamp(args.desde):%Y%m%d}_{pd.Timestamp(args.hasta):%Y%m%d}.csv"
    resultados.to_csv(ruta, index=False)

    total = resultados.groupby("clave")[["coste_total_eur", "coste_energia_eur", "arranques", "toneladas", "energia_mwh"]].sum()
    total["eur_por_t"] = total.coste_total_eur / total.toneladas
    total["precio_medio_eur_mwh"] = total.coste_energia_eur / total.energia_mwh
    total["ahorro_vs_A_eur"] = total.loc["A", "coste_total_eur"] - total.coste_total_eur
    total["ahorro_vs_A_pct"] = 100 * total.ahorro_vs_A_eur / total.loc["A", "coste_total_eur"]
    total.insert(0, "estrategia", resultados.groupby("clave").estrategia.first())
    n = resultados.fecha.nunique()
    print(f"{n} días resueltos, {len(fallos)} sin datos.\n")
    print(_formato(total, COLUMNAS))
    captura = total.loc["B", "ahorro_vs_A_eur"] / total.loc["O", "ahorro_vs_A_eur"]
    por_dia = resultados.pivot(index="fecha", columns="clave", values="coste_total_eur")
    print(f"\nB captura el {100 * captura:.0f} % del ahorro de O frente a A.")
    print(f"Días en que B sale más caro que A: {(por_dia.B > por_dia.A + 0.01).sum()} de {n}.")
    print(f"Resultados por día: {ruta}")
    for fecha, error in fallos[:5]:
        print(f"  sin datos {fecha}: {error}")


HISTORIA_DIAS = 400  # precios previos que necesitan la previsión y el perfil del mes anterior
COLUMNAS_SEMANA = ["estrategia", "coste_total_eur", "arranques", "eur_por_t", "ahorro_vs_A_pct", "captura_pct"]
COLUMNAS_CO2 = ["kgco2_por_t", "emisiones_vs_A_pct"]


def _columnas(tabla):
    return COLUMNAS_SEMANA + [c for c in COLUMNAS_CO2 if c in tabla.columns]


def _futuros(desde, hasta):
    """Futuros de OMIP, solo si se activan con TXANDA_OMIP=1: el proyecto no usa sus datos."""
    if os.environ.get("TXANDA_OMIP") != "1":
        return None

    return tabla_futuros(desde, hasta)


def _matriz(desde, hasta):
    """Matriz diaria de precios, descargando antes lo que falte."""
    for fecha in dias(desde, hasta):
        if fichero_en_cache(fecha) is None:
            precios_omie(fecha)
            time.sleep(0.2)
    matriz = matriz_diaria(desde, hasta)
    return matriz.to_numpy(), pd.DatetimeIndex(matriz.index)


def orden_semana(args):
    from .prevision import ingenua
    from .semana import cargar_semana, simular_semana

    lunes = pd.Timestamp(args.lunes)
    if lunes.weekday() != 0:
        raise SystemExit(f"{lunes:%d-%m-%Y} no es lunes")
    M, fechas = _matriz(lunes - pd.Timedelta(days=HISTORIA_DIAS), lunes + pd.Timedelta(days=6))
    pos = fechas.get_loc(lunes)
    futuros = _futuros(lunes - pd.Timedelta(days=10), lunes + pd.Timedelta(days=5))
    previsores = previsores_semana(lunes, M, fechas, futuros)
    semana = cargar_semana(lunes)
    tabla, planes, errores = simular_semana(Planta(), semana, M, fechas, previsores, fraccion_cara=args.fraccion_cara)
    print(_formato(tabla, _columnas(tabla)))
    print("\nError medio de la previsión (€/MWh) según los días de antelación:")
    print(_formato(errores.groupby("k").mean()))

    def del_domingo(prever):
        partes = [semana.dias[0].to_numpy(dtype=float)]
        partes += [a_cuartos(prever(pos - 1, i + 1), semana.dias[i].index) for i in range(1, 7)]
        return np.concatenate(partes)

    previsiones = {"ingenua": del_domingo(lambda p, k: ingenua(M, p, k))}
    if "F" in previsores:
        previsiones["futuros"] = del_domingo(previsores["F"].predecir)
    previsiones["LEAR"] = del_domingo(previsores["L"].predecir)
    previsiones["ML"] = del_domingo(previsores["D"].predecir)
    ruta = dibujar_semana(Planta(), semana, planes, tabla, SALIDAS / f"semana_{lunes:%Y%m%d}.png", previsiones)
    print(f"Gráfico: {ruta}")


def orden_semanas(args):
    from scipy.stats import wilcoxon

    from .prevision import PrevisorFuturos

    lunes = pd.date_range(args.desde, args.hasta, freq="W-MON")
    M, fechas = _matriz(lunes[0] - pd.Timedelta(days=HISTORIA_DIAS), lunes[-1] + pd.Timedelta(days=6))
    futuros = _futuros(lunes[0] - pd.Timedelta(days=10), lunes[-1] + pd.Timedelta(days=5))
    from .emisiones import descargar
    sin_emisiones = descargar(lunes[0], lunes[-1] + pd.Timedelta(days=6))  # en serie, antes de repartir semanas
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # un hilo por proceso: el paralelismo va por semanas
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        salida = list(pool.map(resolver_semana, lunes, repeat(M), repeat(fechas), repeat(args.fraccion_cara), repeat(futuros)))
    resultados = pd.concat([r for r, _ in salida], ignore_index=True)
    errores = pd.concat([e for _, e in salida], ignore_index=True)
    SALIDAS.mkdir(exist_ok=True)
    sufijo = f"{lunes[0]:%Y%m%d}_{lunes[-1]:%Y%m%d}"
    resultados.to_csv(SALIDAS / f"semanas_{sufijo}.csv", index=False)
    errores.to_csv(SALIDAS / f"semanas_{sufijo}_errores.csv", index=False)

    sumas = ["coste_total_eur", "arranques", "toneladas"] + (["emisiones_t"] if "emisiones_t" in resultados else [])
    total = resultados.groupby("clave", sort=False)[sumas].sum()
    total["eur_por_t"] = total.coste_total_eur / total.toneladas
    base, techo = total.loc["A", "coste_total_eur"], total.loc["O", "coste_total_eur"]
    total["ahorro_vs_A_pct"] = 100 * (base - total.coste_total_eur) / base
    total["captura_pct"] = 100 * (base - total.coste_total_eur) / (base - techo)
    if "emisiones_t" in total:
        total["kgco2_por_t"] = 1000 * total.emisiones_t / total.toneladas
        total["emisiones_vs_A_pct"] = 100 * (total.emisiones_t / total.loc["A", "emisiones_t"] - 1)
    total.insert(0, "estrategia", resultados.groupby("clave", sort=False).estrategia.first())
    print(f"{len(lunes)} semanas, del {lunes[0]:%d-%m-%Y} al {lunes[-1] + pd.Timedelta(days=6):%d-%m-%Y}.\n")
    print(_formato(total, _columnas(total)))
    if "emisiones_t" in total:
        evitadas = (total.loc["A", "emisiones_t"] - total.emisiones_t) * 52 / len(lunes)
        print("\nEmisiones evitadas frente a A (tCO₂-eq al año, planta tipo): "
              + ", ".join(f"{c} {v:,.0f}".replace(",", ".") for c, v in evitadas.drop("A").items()))
    if sin_emisiones:
        print(f"\nDías sin datos de emisiones de REE: {[str(d) for d, _ in sin_emisiones]}")

    semanal = resultados.pivot(index="lunes", columns="clave", values="coste_total_eur")
    print("\nComparaciones semana a semana (diferencia de coste; negativo = la primera es más barata):")
    for a, b in [("H", "F"), ("H", "L"), ("H", "C"), ("H", "OD"), ("F", "C"), ("L", "C"), ("D", "C"), ("L", "F"),
                 ("L", "D"), ("C", "OD"), ("F", "OD"), ("L", "OD"), ("D", "OD"), ("OD", "B"), ("B", "A")]:
        if a not in semanal or b not in semanal:
            continue
        dif = semanal[a] - semanal[b]
        p = wilcoxon(dif).pvalue if (dif != 0).any() else float("nan")
        print(f"  {a} frente a {b}: mediana {dif.median():,.0f} €/semana, {a} gana {(dif < 0).sum()} de {len(dif)} semanas, Wilcoxon p = {p:.3g}".replace(",", "."))
    print("\nError medio de la previsión (€/MWh) según los días de antelación:")
    print(_formato(errores.drop(columns="lunes").groupby("k").mean()))
    if futuros is not None and len(futuros):
        previsor = PrevisorFuturos(M, fechas, futuros)
        pedidas = [(fechas[fechas.get_loc(l) + j - 1], fechas[fechas.get_loc(l) + i]) for l in lunes for j in range(7) for i in range(j + 1, 7)]
        con_cotizacion = sum(previsor.nivel(d, t) is not None for d, t in pedidas)
        print(f"\nFuturos OMIP: {con_cotizacion} de {len(pedidas)} previsiones con cotización; el resto usa la ingenua.")
    print(f"\nResultados: {SALIDAS / f'semanas_{sufijo}.csv'}")


def orden_sensibilidad_arranque(args):
    from scipy.stats import wilcoxon


    lunes = pd.date_range(args.desde, args.hasta, freq="W-MON")
    M, fechas = _matriz(lunes[0] - pd.Timedelta(days=HISTORIA_DIAS), lunes[-1] + pd.Timedelta(days=6))
    futuros = _futuros(lunes[0] - pd.Timedelta(days=10), lunes[-1] + pd.Timedelta(days=5))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        partes = list(pool.map(resolver_semana_sensibilidad, lunes, repeat(M), repeat(fechas), repeat(args.costes),
                               repeat(args.fraccion_cara), repeat(futuros)))
    resultados = pd.concat(partes, ignore_index=True)
    SALIDAS.mkdir(exist_ok=True)
    sufijo = f"{lunes[0]:%Y%m%d}_{lunes[-1]:%Y%m%d}"
    resultados.to_csv(SALIDAS / f"sensibilidad_arranque_{sufijo}.csv", index=False)

    total = resultados.groupby(["coste_arranque_eur", "clave"], sort=False)[["coste_total_eur", "arranques"]].sum()
    coste = total.coste_total_eur.unstack()
    orden = [c for c in ["A", "B", "OD", "C", "F", "L", "D", "O"] if c in coste.columns]
    coste = coste[orden]
    captura = 100 * coste.rsub(coste["A"], axis=0).div(coste["A"] - coste["O"], axis=0)
    ahorro = 100 * coste.rsub(coste["A"], axis=0).div(coste["A"], axis=0)
    arranques = total.arranques.unstack()[orden] / len(lunes)
    n = len(lunes)
    print(f"{n} semanas, del {lunes[0]:%d-%m-%Y} al {lunes[-1] + pd.Timedelta(days=6):%d-%m-%Y}.\n")
    print("Parte del ahorro máximo capturado (%; 0 = A, 100 = O):")
    print(_formato(captura.drop(columns=["A", "O"]).rename_axis("coste arranque")))
    print("\nAhorro del oráculo O frente a A (%), que es el ahorro máximo posible:")
    print(_formato(ahorro[["O"]].rename_axis("coste arranque")))
    print("\nArranques por semana:")
    print(_formato(arranques.rename_axis("coste arranque")))

    print("\nComparaciones semana a semana (gana = semanas en que la primera es más barata; p de Wilcoxon):")
    semanal = resultados.pivot_table(index=["coste_arranque_eur", "lunes"], columns="clave", values="coste_total_eur")
    pares = [p for p in [("F", "OD"), ("L", "OD"), ("F", "C"), ("L", "C"), ("OD", "B"), ("B", "A")] if set(p) <= set(orden)]
    filas = {}
    for c in args.costes:
        bloque = semanal.loc[float(c)]
        filas[c] = {f"{a} vs {b}": f"{((bloque[a] - bloque[b]) < 0).sum()}/{n} · p={wilcoxon(bloque[a] - bloque[b]).pvalue:.2g}" for a, b in pares}
    print(pd.DataFrame.from_dict(filas, orient="index").rename_axis("coste arranque").to_string())

    ruta = dibujar_sensibilidad(captura.drop(columns=["A", "O"]), SALIDAS / f"sensibilidad_arranque_{sufijo}.png",
                                f"Sensibilidad al coste de arranque · {n} semanas ({lunes[0]:%m/%Y}–{lunes[-1]:%m/%Y})")
    print(f"\nGráfico: {ruta}\nResultados: {SALIDAS / f'sensibilidad_arranque_{sufijo}.csv'}")


def orden_sensibilidad_flexibilidad(args):
    from scipy.stats import wilcoxon


    base = Planta()
    variantes = {}
    for rango in args.rangos:
        minimo, maximo = (int(v) for v in rango.split("-"))
        if not minimo <= base.coladas_dia <= maximo:
            raise SystemExit(f"El rango {rango} debe contener las {base.coladas_dia} coladas diarias")
        variantes[rango] = {"holgura_menos": base.coladas_dia - minimo, "holgura_mas": maximo - base.coladas_dia}

    lunes = pd.date_range(args.desde, args.hasta, freq="W-MON")
    M, fechas = _matriz(lunes[0] - pd.Timedelta(days=HISTORIA_DIAS), lunes[-1] + pd.Timedelta(days=6))
    futuros = _futuros(lunes[0] - pd.Timedelta(days=10), lunes[-1] + pd.Timedelta(days=5))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        partes = list(pool.map(resolver_semana_variantes, lunes, repeat(M), repeat(fechas), repeat(variantes),
                               repeat(args.fraccion_cara), repeat(futuros)))
    resultados = pd.concat(partes, ignore_index=True)
    SALIDAS.mkdir(exist_ok=True)
    sufijo = f"{lunes[0]:%Y%m%d}_{lunes[-1]:%Y%m%d}"
    resultados.to_csv(SALIDAS / f"sensibilidad_flexibilidad_{sufijo}.csv", index=False)

    n = len(lunes)
    coste = resultados.groupby(["variante", "clave"], sort=False).coste_total_eur.sum().unstack()
    orden = [c for c in ["A", "B", "OD", "C", "F", "L", "D", "O"] if c in coste.columns]
    coste = coste.loc[list(variantes), orden]
    ahorro = 100 * coste.rsub(coste["A"], axis=0).div(coste["A"], axis=0)
    captura = 100 * coste.rsub(coste["A"], axis=0).div(coste["A"] - coste["O"], axis=0)
    anual = coste.rsub(coste["A"], axis=0) * 52 / n / 1000
    print(f"{n} semanas, del {lunes[0]:%d-%m-%Y} al {lunes[-1] + pd.Timedelta(days=6):%d-%m-%Y}. "
          f"Rango = coladas mínimas y máximas por día; el total semanal es siempre {7 * base.coladas_dia}.\n")
    print("Ahorro frente al horario fijo A (%):")
    print(_formato(ahorro.drop(columns="A").rename_axis("rango diario")))
    print("\nAhorro frente a A (k€ al año, planta tipo):")
    print(_formato(anual.drop(columns="A").rename_axis("rango diario")))
    print("\nParte del ahorro máximo capturado (%; 0 = A, 100 = O):")
    print(_formato(captura.drop(columns=["A", "O"]).rename_axis("rango diario")))

    print("\nComparaciones semana a semana (gana = semanas en que la primera es más barata; p de Wilcoxon):")
    semanal = resultados.pivot_table(index=["variante", "lunes"], columns="clave", values="coste_total_eur")
    pares = [p for p in [("F", "OD"), ("L", "OD"), ("F", "C"), ("L", "C")] if set(p) <= set(orden)]
    filas = {}
    for etiqueta in variantes:
        bloque = semanal.loc[etiqueta]
        filas[etiqueta] = {}
        for a, b in pares:
            dif = bloque[a] - bloque[b]
            p = wilcoxon(dif).pvalue if (dif != 0).any() else float("nan")
            filas[etiqueta][f"{a} vs {b}"] = f"{(dif < 0).sum()}/{n} · p={p:.2g}"
    print(pd.DataFrame.from_dict(filas, orient="index").rename_axis("rango diario").to_string())

    ruta = dibujar_sensibilidad(
        ahorro.drop(columns=["A", "O"]), SALIDAS / f"sensibilidad_flexibilidad_{sufijo}.png",
        f"Sensibilidad a la flexibilidad diaria · {n} semanas ({lunes[0]:%m/%Y}–{lunes[-1]:%m/%Y})",
        etiquetas_x=[f"{r.replace('-', '–')} coladas" for r in variantes],
        eje_x="coladas permitidas cada día (126 a la semana)", eje_y="% de ahorro frente al horario fijo A",
        techo=ahorro["O"],
    )
    print(f"\nGráfico: {ruta}\nResultados: {SALIDAS / f'sensibilidad_flexibilidad_{sufijo}.csv'}")


def orden_publicar(args):
    """Copia a resultados/ una versión de los resultados sin las estrategias con futuros de OMIP (F y H),
    que no se pueden publicar sin su autorización. Es lo que enseña la app publicada."""
    destino = Path(__file__).resolve().parent.parent / "resultados"
    destino.mkdir(exist_ok=True)
    sin_omip = ["F", "H"]
    semanas = pd.read_csv(sorted(f for f in SALIDAS.glob("semanas_*.csv") if not f.stem.endswith("_errores"))[-1])
    semanas[~semanas.clave.isin(sin_omip)].to_csv(destino / "semanas.csv", index=False)
    for nombre, indice, kwargs in [
        ("sensibilidad_arranque", "coste_arranque_eur", {}),
        ("sensibilidad_flexibilidad", "variante", {}),
    ]:
        datos = pd.read_csv(sorted(SALIDAS.glob(f"{nombre}_*.csv"))[-1])
        coste = datos.groupby([indice, "clave"], sort=False).coste_total_eur.sum().unstack()
        n = datos.lunes.nunique()
        columnas = [c for c in ["B", "OD", "C", "L", "D"] if c in coste.columns]
        if nombre == "sensibilidad_arranque":
            valores = 100 * coste.rsub(coste["A"], axis=0).div(coste["A"] - coste["O"], axis=0)
            dibujar_sensibilidad(valores[columnas], destino / f"{nombre}.png",
                                 f"Sensibilidad al coste de arranque · {n} semanas (10/2025–09/2026)")
        else:
            valores = 100 * coste.rsub(coste["A"], axis=0).div(coste["A"], axis=0)
            dibujar_sensibilidad(valores[columnas], destino / f"{nombre}.png",
                                 f"Sensibilidad a la flexibilidad diaria · {n} semanas (10/2025–09/2026)",
                                 etiquetas_x=[f"{r.replace('-', '–')} coladas" for r in valores.index],
                                 eje_x="coladas permitidas cada día (126 a la semana)",
                                 eje_y="% de ahorro frente al horario fijo A", techo=valores["O"])
    print(f"Resultados públicos en {destino}: {', '.join(sorted(p.name for p in destino.iterdir()))}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="txanda")
    parser.add_argument("--fraccion-cara", type=float, default=0.25, help="parte del día que la regla B considera cara")
    sub = parser.add_subparsers(required=True)
    p_dia = sub.add_parser("dia", help="resuelve un día y dibuja el gráfico")
    p_dia.add_argument("fecha")
    p_dia.set_defaults(func=orden_dia)
    p_per = sub.add_parser("periodo", help="backtest día a día entre dos fechas")
    p_per.add_argument("desde")
    p_per.add_argument("hasta")
    p_per.add_argument("--procesos", type=int, default=os.cpu_count(), help="días que se resuelven en paralelo")
    p_per.set_defaults(func=orden_periodo)
    p_sem = sub.add_parser("semana", help="simula una semana con horizonte rodante y dibuja el gráfico")
    p_sem.add_argument("lunes")
    p_sem.set_defaults(func=orden_semana)
    p_sems = sub.add_parser("semanas", help="backtest de todas las semanas entre dos lunes")
    p_sems.add_argument("desde")
    p_sems.add_argument("hasta")
    p_sems.add_argument("--procesos", type=int, default=os.cpu_count(), help="semanas que se resuelven en paralelo")
    p_sems.set_defaults(func=orden_semanas)
    p_sens = sub.add_parser("sensibilidad-arranque", help="backtest semanal con varios costes de arranque")
    p_sens.add_argument("desde")
    p_sens.add_argument("hasta")
    p_sens.add_argument("--costes", type=float, nargs="+", default=[0, 1000, 2500, 5000, 10000])
    p_sens.add_argument("--procesos", type=int, default=os.cpu_count())
    p_sens.set_defaults(func=orden_sensibilidad_arranque)
    p_flex = sub.add_parser("sensibilidad-flexibilidad", help="backtest semanal con varios rangos de coladas diarias")
    p_flex.add_argument("desde")
    p_flex.add_argument("hasta")
    p_flex.add_argument("--rangos", nargs="+", default=["18-18", "16-20", "14-22", "12-22", "8-22", "0-22"],
                        help="coladas mínimas-máximas por día; 12-22 es el caso base")
    p_flex.add_argument("--procesos", type=int, default=os.cpu_count())
    p_flex.set_defaults(func=orden_sensibilidad_flexibilidad)
    p_pub = sub.add_parser("publicar", help="prepara resultados/ para la app publicada, sin datos de OMIP")
    p_pub.set_defaults(func=orden_publicar)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
