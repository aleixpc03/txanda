"""Simulación de una semana (lunes a domingo) con horizonte rodante.

Cada día D, a las 12:00, se conocen los precios de D+1. Las estrategias que deciden
día a día resuelven el resto de la semana con lo que saben en ese momento, fijan las
coladas que empiezan en D+1 y repiten al día siguiente.

    A   Horario fijo: el mejor para el perfil del mes anterior. 18 coladas cada día.
    B   Paradas en horas caras de D+1. 18 coladas cada día.
    OD  Óptimo diario: precio exacto de D+1. 18 coladas cada día.
    C   Semanal con previsión ingenua de D+2…domingo. 126 coladas a la semana, 12–22 al día.
    F   Semanal con el nivel diario de los futuros de OMIP y la forma de la semana anterior.
    L   Semanal con previsión LEAR (LASSO autorregresivo). Mismos cupos que C.
    H   Semanal con LEAR a dos días y futuros de OMIP desde tres. Mismos cupos que C.
    D   Semanal con previsión de aprendizaje automático. Mismos cupos que C.
    LR  Como L, pero el día en que no se puede usar la previsión (faltan datos o no se puede
        calcular) vuelve al óptimo diario. Mismos cupos que C.
    O   Oráculo: conoce toda la semana de antemano. Mismos cupos que C. No es implementable.

Todas producen lo mismo y se evalúan igual: energía a precio real más arranques.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estrategias import DESEMPATE_EUR, coste_colada, disponibilidad, marcar_caros
from .milp import Cupo, Plan, evaluar, resolver, verificar_plan
from .planta import Planta
from .precios import a_cuartos, perfil_mes_anterior, precios_omie
from .prevision import ingenua

NOMBRES_SEMANA = {
    "A": "Horario fijo (mes anterior)",
    "B": "Paradas en horas caras",
    "OD": "Óptimo diario",
    "C": "Semanal, previsión ingenua",
    "F": "Semanal, futuros OMIP",
    "L": "Semanal, previsión LEAR",
    "H": "Semanal, híbrido LEAR + futuros",
    "D": "Semanal, previsión ML",
    "LR": "Semanal, LEAR con vuelta al óptimo diario",
    "O": "Oráculo semanal",
}


@dataclass
class Semana:
    dias: list[pd.Series]  # precio real de cada día, en sus cuartos de hora reales

    @property
    def lunes(self):
        return self.dias[0].index[0].date()

    @property
    def precios(self) -> np.ndarray:
        return np.concatenate([d.to_numpy(dtype=float) for d in self.dias])

    @property
    def cuartos(self) -> pd.DatetimeIndex:
        return self.dias[0].index.append([d.index for d in self.dias[1:]])

    @property
    def limites(self) -> np.ndarray:
        """Primer cuarto de cada día en el eje semanal, más el final."""
        return np.cumsum([0] + [len(d) for d in self.dias])


def cargar_semana(lunes) -> Semana:
    return Semana([precios_omie(f) for f in pd.date_range(lunes, periods=7)])


def intensidad_semana(semana: Semana) -> np.ndarray | None:
    """Intensidad de emisiones en los cuartos de la semana; None si faltan datos de REE."""
    from .emisiones import intensidad_dia

    try:
        return np.concatenate([intensidad_dia(d.index[0].date(), d.index) for d in semana.dias])
    except Exception:
        return None


def cupos_diarios(planta: Planta, semana: Semana) -> list[Cupo]:
    lim = semana.limites
    return [Cupo(lim[i], lim[i + 1], planta.coladas_dia, planta.coladas_dia) for i in range(7)]


def cupos_semanales(planta: Planta, semana: Semana) -> list[Cupo]:
    lim = semana.limites
    diarios = [Cupo(lim[i], lim[i + 1], planta.coladas_dia_min, planta.coladas_dia_max) for i in range(7)]
    return diarios + [Cupo(0, lim[-1], 7 * planta.coladas_dia, 7 * planta.coladas_dia)]


# Tolerancia relativa del solver. 0 = óptimo demostrado (caso base); en escenarios más difíciles se
# puede relajar a 1e-4 (0,01 % del coste de la semana) desde la línea de órdenes.
GAP = 0.0


def _resolver(planta, senal, disponible, cupos, fijadas=None) -> Plan:
    c = coste_colada(planta, senal) + DESEMPATE_EUR * np.arange(len(senal))
    return resolver(planta, c, planta.coste_arranque_eur, disponible, cupos, fijadas, gap=GAP)


def _rodar(planta: Planta, semana: Semana, disponible, cupos, senal_paso, dias_vista: int = 7) -> Plan:
    """Decide día a día: en el paso j fija las coladas que empiezan el día j.

    `dias_vista` limita el modelo a los días j … j + dias_vista − 1. Con cupos diarios
    basta con mirar un día más allá, para las secuencias que cruzan medianoche.
    """
    lim, comprometidas, segundos, estado = semana.limites, (), 0.0, "Optimal"
    for j in range(7):
        fin = lim[min(7, j + dias_vista)]
        cupos_vista = [c for c in cupos if c.fin <= fin]
        plan = _resolver(planta, senal_paso(j)[:fin], disponible[:fin], cupos_vista, fijadas=(lim[j], comprometidas))
        segundos += plan.segundos
        estado = estado if plan.estado == "Optimal" else plan.estado
        comprometidas = tuple(s for s in plan.inicios if s < lim[j + 1])
    return Plan(comprometidas, estado, segundos)


def reparto(quedan: int, dias: int) -> list[int]:
    """Reparte `quedan` coladas entre `dias` días a partes iguales; las que sobran, a los primeros."""
    base, resto = divmod(quedan, dias)
    return [base + (i < resto) for i in range(dias)]


def _rodar_con_respaldo(planta: Planta, semana: Semana, disponible, cupos, senal_paso, senal_diaria, motivo_paso) -> Plan:
    """Como `_rodar` con cupos semanales, pero vuelve al óptimo diario el día en que
    `motivo_paso(j)` devuelve un motivo.

    Ese día no se usa ninguna previsión: las coladas que quedan de la semana se reparten a
    partes iguales entre los días que faltan y las de mañana se colocan con su precio real,
    mirando un día más allá para las secuencias que cruzan medianoche, como el óptimo diario.
    Al día siguiente se vuelve a comprobar la previsión.
    """
    lim, comprometidas, segundos, estado, respaldo = semana.limites, (), 0.0, "Optimal", []
    for j in range(7):
        motivo = motivo_paso(j)
        if motivo is None:
            fin, senal, cupos_paso = lim[7], senal_paso(j), list(cupos)
        else:
            respaldo.append((j, motivo))
            fin, senal = lim[min(7, j + 2)], senal_diaria(j)
            partes = reparto(7 * planta.coladas_dia - len(comprometidas), 7 - j)
            cupos_paso = [Cupo(lim[j + i], lim[j + i + 1], n, n) for i, n in enumerate(partes) if lim[j + i + 1] <= fin]
        plan = _resolver(planta, senal[:fin], disponible[:fin], cupos_paso, fijadas=(lim[j], comprometidas))
        segundos += plan.segundos
        estado = estado if plan.estado == "Optimal" else plan.estado
        comprometidas = tuple(s for s in plan.inicios if s < lim[j + 1])
    return Plan(comprometidas, estado, segundos, tuple(respaldo))


def simular_semana(
    planta: Planta,
    semana: Semana,
    M: np.ndarray,
    fechas: pd.DatetimeIndex,
    previsores: dict | None = None,
    perfil: np.ndarray | None = None,
    fraccion_cara: float = 0.25,
    intensidad: np.ndarray | None = None,
    vigilancias: dict | None = None,
):
    """Resuelve las estrategias. Devuelve tabla, planes y errores de previsión.

    M y fechas son la matriz diaria (96 cuartos por hora de reloj) con al menos dos
    semanas antes del lunes. `previsores` asocia cada estrategia semanal con previsión
    entrenada a su previsor, por ejemplo {"L": PrevisorLEAR, "D": PrevisorML}. Con
    `intensidad` (tCO₂-eq/MWh en los cuartos de la semana) se añaden las emisiones.
    `vigilancias` asocia una estrategia de `previsores` con su Vigilancia, por ejemplo
    {"L": Vigilancia}, y añade su versión con vuelta al óptimo diario («LR»).
    """
    real = semana.precios
    lim, T = semana.limites, len(real)
    pos_lunes = fechas.get_loc(pd.Timestamp(semana.lunes))
    disponible = disponibilidad(planta, semana.cuartos)
    diarios, semanales = cupos_diarios(planta, semana), cupos_semanales(planta, semana)
    if perfil is None:
        perfil = perfil_mes_anterior(semana.lunes)

    def tramo(i):
        return slice(lim[i], lim[i + 1])

    def con_futuro(j, prever):
        """Precio real hasta el día j (D+1) y la previsión hecha en D para los días siguientes."""
        senal = real.copy()
        for i in range(j + 1, 7):
            senal[tramo(i)] = 0.0 if prever is None else a_cuartos(prever(pos_lunes + j - 1, i - j + 1), semana.dias[i].index)
        return senal

    def regla(j):
        senal = np.zeros(T)
        dia = real[tramo(j)]
        caros = marcar_caros(dia, fraccion_cara)
        prima = dia[caros].mean() - dia[~caros].mean() if caros.any() and (~caros).any() else 0.0
        senal[tramo(j)] = prima * caros
        return senal

    senal_fija = np.concatenate([a_cuartos(perfil, d.index) for d in semana.dias])
    planes = {
        "A": _resolver(planta, senal_fija, disponible, diarios),
        "B": _rodar(planta, semana, disponible, diarios, regla, dias_vista=2),
        "OD": _rodar(planta, semana, disponible, diarios, lambda j: con_futuro(j, None), dias_vista=2),
        "C": _rodar(planta, semana, disponible, semanales, lambda j: con_futuro(j, lambda p, k: ingenua(M, p, k))),
    }
    previsores = previsores or {}
    for clave, previsor in previsores.items():
        planes[clave] = _rodar(planta, semana, disponible, semanales, lambda j, p=previsor: con_futuro(j, p.predecir))
    for clave, vigilancia in (vigilancias or {}).items():
        planes[f"{clave}R"] = _rodar_con_respaldo(
            planta, semana, disponible, semanales,
            lambda j, p=previsores[clave]: con_futuro(j, p.predecir),
            lambda j: con_futuro(j, None),
            lambda j, v=vigilancia: v.motivo(pos_lunes + j - 1, range(2, 8 - j)),
        )
    planes["O"] = _resolver(planta, real, disponible, semanales)

    filas = {}
    for clave, plan in planes.items():
        verificar_plan(planta, disponible, plan.inicios, diarios if clave in ("A", "B", "OD") else semanales)
        nombre = NOMBRES_SEMANA.get(clave) or f"{NOMBRES_SEMANA[clave[:-1]]}, con vuelta al óptimo diario"
        filas[clave] = {"estrategia": nombre, **evaluar(planta, real, plan.inicios, intensidad),
                        "segundos": plan.segundos, "optimo_demostrado": plan.estado == "Optimal",
                        "dias_respaldo": len(plan.respaldo)}
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    base, techo = tabla.loc["A", "coste_total_eur"], tabla.loc["O", "coste_total_eur"]
    tabla["ahorro_vs_A_eur"] = base - tabla["coste_total_eur"]
    tabla["ahorro_vs_A_pct"] = 100 * tabla["ahorro_vs_A_eur"] / base
    tabla["captura_pct"] = 100 * tabla["ahorro_vs_A_eur"] / (base - techo) if base > techo else np.nan
    if intensidad is not None:
        tabla["emisiones_vs_A_pct"] = 100 * (tabla["emisiones_t"] / tabla.loc["A", "emisiones_t"] - 1)

    errores = []
    for j in range(7):
        pos_D = pos_lunes + j - 1
        for i in range(j + 1, 7):
            k, objetivo = i - j + 1, M[pos_lunes + i]
            fila = {"k": k, "mae_ingenua": np.abs(ingenua(M, pos_D, k) - objetivo).mean()}
            for clave, previsor in previsores.items():
                fila[f"mae_{clave}"] = np.abs(previsor.predecir(pos_D, k) - objetivo).mean()
            errores.append(fila)
    return tabla, planes, pd.DataFrame(errores)
