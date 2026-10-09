"""MILP de programación de coladas y comprobación independiente de planes.

El horizonte puede ser un día o una semana: t recorre sus cuartos de hora en orden y las
secuencias pueden cruzar medianoche. Los cupos fijan cuántas coladas empiezan en cada
tramo (por ejemplo, cada día). Las coladas ya comprometidas se pasan como fijadas.

Formulación por secuencias (t = cuarto de hora, d = duración de una colada,
G = cambio de artesa, Lmin..Lmax = coladas por secuencia):

    y[s,L] ∈ {0,1}   empieza en s una secuencia de L coladas seguidas; ocupa el horno
                     en [s, s + L·d) y deja el cambio de artesa en [s + L·d, s + L·d + G)

    min  Σ y[s,L]·( Σ_{j<L} c[s + j·d] + w )
    s.a. un camino de 0 a T: en cada cuarto t el horno pasa a t+1 parado (u[t]) o
         empieza una secuencia y salta a s + L·d + G     (horno y artesa, sin solapes)
         mín_k ≤ Σ y[s,L]·n_k(s,L) ≤ máx_k            ∀k   (cupos: n_k = coladas de la
                                                            secuencia que empiezan en el tramo k)
         y[s,L] = 0 si alguna colada no cabe en el horizonte o cae fuera de turno
         las secuencias ya comprometidas se mantienen; si siguen abiertas en el
         momento de decidir, solo se elige cuánto se alargan

Sin los cupos sería un camino mínimo en un grafo acíclico, con relajación lineal entera;
los cupos lo rompen muy poco, y una semana (≈ 4.500 secuencias posibles) se resuelve en
segundos. Cada
estrategia solo cambia c y w: la planta y las restricciones son las mismas.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pulp

from .planta import DT_H, Planta


@dataclass(frozen=True)
class Cupo:
    """Entre `minimo` y `maximo` coladas deben empezar en los cuartos [inicio, fin)."""

    inicio: int
    fin: int
    minimo: int
    maximo: int


def cupo_unico(planta: Planta, T: int) -> list[Cupo]:
    """Un solo día: exactamente coladas_dia en todo el horizonte."""
    return [Cupo(0, T, planta.coladas_dia, planta.coladas_dia)]


@dataclass(frozen=True)
class Plan:
    inicios: tuple[int, ...]  # cuarto de hora en el que empieza cada colada
    estado: str
    segundos: float
    respaldo: tuple[tuple[int, str], ...] = ()  # (día de la semana, motivo) en que se volvió al óptimo diario


def inicios_permitidos(planta: Planta, disponible: np.ndarray) -> np.ndarray:
    """Cuartos en los que puede empezar una colada: cabe entera en el día y en horario con personal."""
    d, T = planta.duracion, len(disponible)
    permitido = np.zeros(T, dtype=bool)
    for s in range(T - d + 1):
        permitido[s] = disponible[s : s + d].all()
    return permitido


def _coladas_seguidas(permitido: np.ndarray, s: int, d: int, tope: int) -> int:
    """Cuántas coladas seguidas caben desde s, hasta un máximo de `tope`."""
    n = 0
    while n < tope and s + n * d < len(permitido) and permitido[s + n * d]:
        n += 1
    return n


def _coladas_en_tramo(s: int, L: int, d: int, inicio: int, fin: int) -> int:
    """Coladas de la secuencia (s, L) que empiezan en [inicio, fin)."""
    primera = max(0, -(-(inicio - s) // d))
    ultima = min(L, -(-(fin - s) // d))
    return max(0, ultima - primera)


def _entera(variables, tol: float = 1e-6) -> bool:
    valores = np.array([v.value() for v in variables], dtype=float)
    return bool(np.all(np.minimum(np.abs(valores), np.abs(1 - valores)) <= tol))


def resolver(
    planta: Planta,
    coste_inicio: np.ndarray,
    peso_arranque: float,
    disponible: np.ndarray,
    cupos: list[Cupo] | None = None,
    fijadas: tuple[int, tuple[int, ...]] | None = None,
    limite_s: float = 60.0,
    gap: float = 0.0,
) -> Plan:
    """Resuelve el MILP con coste c[s] por colada que empieza en s y peso w por secuencia.

    `fijadas = (hasta, inicios)`: las coladas que empiezan antes del cuarto `hasta`
    están ya decididas y son exactamente `inicios`.
    """
    d, G = planta.duracion, planta.cambio_secuencia_cuartos
    Lmin, Lmax = planta.secuencia_min, planta.secuencia_max
    T = len(disponible)
    permitido = inicios_permitidos(planta, disponible)
    cupos = cupos if cupos is not None else cupo_unico(planta, T)
    hasta, comprometidas = fijadas if fijadas is not None else (0, ())
    if any(s >= hasta or not permitido[s] for s in comprometidas):
        raise ValueError("Hay coladas comprometidas después de `hasta` o en cuartos no permitidos")

    # Secuencias ya empezadas: si la siguiente colada caería antes de `hasta`, está
    # cerrada y su longitud es fija; si no, sigue abierta y puede alargarse.
    obligadas = {}
    for cadena in secuencias(planta, comprometidas):
        abierta = cadena[-1] + d >= hasta
        obligadas[cadena[0]] = (max(len(cadena), Lmin), Lmax) if abierta else (len(cadena), len(cadena))

    candidatas = []
    for s in np.flatnonzero(permitido).tolist():
        if s < hasta and s not in obligadas:
            continue
        minimo, maximo = obligadas.get(s, (Lmin, Lmax))
        maximo = min(maximo, _coladas_seguidas(permitido, s, d, maximo))
        candidatas += [(s, L) for L in range(minimo, maximo + 1)]

    prob = pulp.LpProblem("coladas", pulp.LpMinimize)
    y = {(s, L): prob.add_variable(f"y_{s}_{L}", cat="Binary") for s, L in candidatas}
    coste = {}
    for s, L in candidatas:
        coste[s, L] = float(sum(coste_inicio[s + j * d] for j in range(L))) + peso_arranque
    prob += pulp.lpSum(coste[k] * y[k] for k in candidatas)

    for s0 in obligadas:
        opciones = [y[k] for k in candidatas if k[0] == s0]
        if not opciones:
            raise RuntimeError(f"La secuencia comprometida del cuarto {s0} no puede completarse")
        prob += pulp.lpSum(opciones) == 1, f"comprometida_{s0}"

    # Camino por el tiempo: cada cuarto es un nodo; el horno avanza de t a t+1 parado
    # o salta con una secuencia hasta el final de su cambio de artesa.
    espera = [prob.add_variable(f"u_{t}", lowBound=0, upBound=1) for t in range(T)]
    salen: dict[int, list] = {t: [espera[t]] for t in range(T)}
    entran: dict[int, list] = {t: [espera[t - 1]] for t in range(1, T + 1)}
    for s, L in candidatas:
        salen[s].append(y[s, L])
        entran[min(T, s + L * d + G)].append(y[s, L])
    for t in range(T + 1):
        balance = pulp.lpSum(entran.get(t, [])) - pulp.lpSum(salen.get(t, []))
        prob += balance == (-1 if t == 0 else 1 if t == T else 0), f"camino_{t}"

    for k, cupo in enumerate(cupos):
        expr = pulp.lpSum(
            n * y[s, L] for s, L in candidatas if (n := _coladas_en_tramo(s, L, d, cupo.inicio, cupo.fin))
        )
        if cupo.minimo == cupo.maximo:
            prob += expr == cupo.minimo, f"cupo_{k}"
        else:
            prob += expr >= cupo.minimo, f"cupo_min_{k}"
            prob += expr <= cupo.maximo, f"cupo_max_{k}"

    # Primero la relajación lineal: si ya es entera, es el óptimo del MILP y cuesta unas
    # cinco veces menos (pasa en ~8 de cada 10 modelos). Si no, el MILP completo.
    inicio = time.perf_counter()
    resultado = prob.solve(pulp.HiGHS(msg=False, mip=False, timeLimit=limite_s))
    if resultado.status != pulp.LpSolveStatus.Optimal or not _entera(y.values()):
        resultado = prob.solve(pulp.HiGHS(msg=False, timeLimit=limite_s, gapRel=gap))
    segundos = time.perf_counter() - inicio
    estado = resultado.status.name
    # Si se agota el tiempo pero hay solución, se usa la mejor encontrada y queda anotado en `estado`.
    con_solucion = resultado.status == pulp.LpSolveStatus.TimeLimit and resultado.has_solution
    if resultado.status != pulp.LpSolveStatus.Optimal and not con_solucion:
        raise RuntimeError(
            f"El MILP no tiene solución óptima ({estado}). Revisa cupos, turnos y "
            "secuencias: puede que las coladas no quepan en el horizonte."
        )
    inicios = tuple(sorted(s + j * d for (s, L), v in y.items() if v.value() > 0.5 for j in range(L)))
    return Plan(inicios=inicios, estado=estado, segundos=segundos)


def secuencias(planta: Planta, inicios) -> list[list[int]]:
    """Agrupa las coladas en secuencias: dos coladas seguidas sin hueco comparten secuencia."""
    grupos: list[list[int]] = []
    for s in sorted(inicios):
        if grupos and s == grupos[-1][-1] + planta.duracion:
            grupos[-1].append(s)
        else:
            grupos.append([s])
    return grupos


def verificar_plan(planta: Planta, disponible: np.ndarray, inicios, cupos: list[Cupo] | None = None) -> None:
    """Comprueba un plan sin pasar por el MILP. Lanza ValueError con todas las violaciones."""
    d, T = planta.duracion, len(disponible)
    permitido = inicios_permitidos(planta, disponible)
    orden = sorted(inicios)
    errores = []
    for cupo in cupos if cupos is not None else cupo_unico(planta, T):
        n = sum(cupo.inicio <= s < cupo.fin for s in orden)
        if not cupo.minimo <= n <= cupo.maximo:
            errores.append(f"{n} coladas entre los cuartos {cupo.inicio} y {cupo.fin}, fuera de [{cupo.minimo}, {cupo.maximo}]")
    for s in orden:
        if not (0 <= s < T and permitido[s]):
            errores.append(f"la colada del cuarto {s} no cabe en el horizonte o cae fuera de turno")
    for a, b in zip(orden, orden[1:]):
        if b - a < d:
            errores.append(f"coladas solapadas en los cuartos {a} y {b}")
    grupos = secuencias(planta, orden)
    for g in grupos:
        if not planta.secuencia_min <= len(g) <= planta.secuencia_max:
            errores.append(f"secuencia de {len(g)} coladas desde el cuarto {g[0]}")
    for anterior, siguiente in zip(grupos, grupos[1:]):
        hueco = siguiente[0] - (anterior[-1] + d)
        if 0 <= hueco < planta.cambio_secuencia_cuartos:
            errores.append(f"cambio de artesa de {hueco} cuartos antes del cuarto {siguiente[0]}")
    if errores:
        raise ValueError("Plan no válido: " + "; ".join(errores))


def potencia(planta: Planta, inicios, T: int) -> np.ndarray:
    """Potencia del horno en cada cuarto de hora (MW)."""
    p = np.zeros(T)
    for s in inicios:
        p[s : s + planta.duracion] += planta.perfil_mw
    return p


def evaluar(planta: Planta, precios: np.ndarray, inicios, intensidad: np.ndarray | None = None) -> dict:
    """Coste real de un plan con los precios del día, igual para todas las estrategias.

    Con `intensidad` (tCO₂-eq/MWh por cuarto de hora) añade las emisiones asociadas.
    """
    energia = potencia(planta, inicios, len(precios)) * DT_H
    arranques = len(secuencias(planta, inicios))
    coste_energia = float(energia @ precios)
    coste_arranques = arranques * planta.coste_arranque_eur
    toneladas = len(inicios) * planta.toneladas_colada
    total = coste_energia + coste_arranques
    resultado = {
        "coladas": len(inicios),
        "toneladas": toneladas,
        "energia_mwh": float(energia.sum()),
        "precio_medio_eur_mwh": coste_energia / energia.sum() if energia.sum() else float("nan"),
        "coste_energia_eur": coste_energia,
        "arranques": arranques,
        "coste_arranques_eur": coste_arranques,
        "coste_total_eur": total,
        "eur_por_t": total / toneladas if toneladas else float("nan"),
    }
    if intensidad is not None:
        resultado["emisiones_t"] = float(energia @ intensidad)
        resultado["kgco2_por_t"] = 1000 * resultado["emisiones_t"] / toneladas if toneladas else float("nan")
    return resultado
