import numpy as np
import pandas as pd

from txanda.estrategias import coste_colada, disponibilidad
from txanda.milp import secuencias
from txanda.planta import Planta
from txanda.precios import vector_dia
from txanda.semana import Semana, _resolver, _rodar, cupos_semanales, simular_semana
from tests.test_milp_dia import dia_sintetico

LUNES = pd.Timestamp("2026-03-02")


def datos_sinteticos():
    """Semana sintética del 2 al 8 de marzo de 2026 y matriz con tres semanas previas."""
    fechas = pd.date_range(LUNES - pd.Timedelta(days=21), periods=28)
    dias = [dia_sintetico(f, semilla=i) + 15 * np.sin(i) for i, f in enumerate(fechas)]
    M = np.vstack([vector_dia(d) for d in dias])
    return Semana(dias[21:]), M, fechas


def test_semana_cumple_cupos_y_el_oraculo_es_el_mejor():
    semana, M, fechas = datos_sinteticos()
    planta = Planta()
    tabla, planes, errores = simular_semana(planta, semana, M, fechas, perfil=M[:7].mean(axis=0))  # verifica cada plan
    assert set(tabla.index) == {"A", "B", "OD", "C", "O"}
    assert (tabla.coladas == 7 * planta.coladas_dia).all()
    assert (tabla.coste_total_eur >= tabla.loc["O", "coste_total_eur"] - 1e-6).all()
    assert tabla.loc["O", "captura_pct"] == 100 and tabla.loc["A", "captura_pct"] == 0
    assert sorted(errores.k.unique()) == list(range(2, 8))


def test_rodar_con_informacion_perfecta_reproduce_el_oraculo():
    semana, _, _ = datos_sinteticos()
    planta = Planta()
    disponible = disponibilidad(planta, semana.cuartos)
    cupos = cupos_semanales(planta, semana)
    oraculo = _resolver(planta, semana.precios, disponible, cupos)
    rodante = _rodar(planta, semana, disponible, cupos, lambda j: semana.precios)
    coste = lambda plan: coste_colada(planta, semana.precios)[list(plan.inicios)].sum() + planta.coste_arranque_eur * len(secuencias(planta, plan.inicios))
    assert abs(coste(rodante) - coste(oraculo)) < 1.0
