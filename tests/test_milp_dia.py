import itertools

import numpy as np
import pandas as pd
import pytest

from txanda.estrategias import comparar_dia, disponibilidad, plan_fijo, plan_optimo, plan_regla
from txanda.milp import evaluar, potencia, secuencias, verificar_plan
from txanda.planta import Planta
from txanda.precios import cuartos_del_dia, precios_omie


def dia_sintetico(fecha="2026-02-24", semilla=0):
    """Curva con valle solar negativo a mediodía y pico de tarde, más ruido."""
    cuartos = cuartos_del_dia(fecha)
    h = cuartos.hour + cuartos.minute / 60
    rng = np.random.default_rng(semilla)
    p = 90 - 110 * np.exp(-((h - 13.5) ** 2) / 6) + 90 * np.exp(-((h - 20.5) ** 2) / 2) + rng.normal(0, 6, len(h))
    return pd.Series(p, index=cuartos)


@pytest.mark.parametrize("fecha,n", [("2026-09-25", 96), ("2026-03-29", 92), ("2026-10-25", 100)])
def test_cuartos_del_dia_con_cambio_de_hora(fecha, n):
    assert len(cuartos_del_dia(fecha)) == n


def test_omie_horario_se_expande_y_usa_la_columna_de_espana(tmp_path):
    lineas = ["MARGINALPDBC;"] + [f"2025;09;15;{h};{h + 1000};{h};" for h in range(24, 0, -1)] + ["*"]
    (tmp_path / "marginalpdbc_20250915.1").write_text("\n".join(lineas), encoding="latin-1")
    precios = precios_omie("2025-09-15", cache=tmp_path, cache_ree=None)
    assert len(precios) == 96
    assert precios.iloc[:4].tolist() == [1.0] * 4  # periodo 1 → España, repetido en 4 cuartos
    assert precios.iloc[-1] == 24.0


@pytest.mark.parametrize("semilla", range(6))
def test_optimo_coincide_con_fuerza_bruta(semilla):
    planta = Planta(perfil_mw=(10.0, 5.0), coladas_dia=4, secuencia_min=2, secuencia_max=3,
                    cambio_secuencia_cuartos=1, coste_arranque_eur=7.0)
    T = 20
    precios = np.random.default_rng(semilla).normal(50, 30, T)
    disponible = np.ones(T, dtype=bool)

    mejor = np.inf
    for inicios in itertools.combinations(range(T - planta.duracion + 1), planta.coladas_dia):
        try:
            verificar_plan(planta, disponible, inicios)
        except ValueError:
            continue
        mejor = min(mejor, evaluar(planta, precios, inicios)["coste_total_eur"])

    plan = plan_optimo(planta, precios, disponible)
    verificar_plan(planta, disponible, plan.inicios)
    assert evaluar(planta, precios, plan.inicios)["coste_total_eur"] == pytest.approx(mejor, abs=1e-6)


def perfil_de(precios):
    """Perfil «del mes anterior» para los tests: el de otro día sintético."""
    return dia_sintetico(semilla=99).to_numpy()[: len(precios)]


@pytest.mark.parametrize("semilla", range(3))
def test_estrategias_validas_y_optimo_no_pierde(semilla):
    precios = dia_sintetico(semilla=semilla)
    tabla, _ = comparar_dia(Planta(), precios, perfil_de(precios))  # comparar_dia verifica cada plan
    assert (tabla.coladas == Planta().coladas_dia).all()
    assert tabla.loc["O", "coste_total_eur"] <= tabla.loc["A", "coste_total_eur"] + 1e-6
    assert tabla.loc["O", "coste_total_eur"] <= tabla.loc["B", "coste_total_eur"] + 1e-6


def test_horario_fijo_con_perfil_plano_arranca_al_abrir_y_llena_la_artesa():
    planta = Planta()
    plan = plan_fijo(planta, np.zeros(96), np.ones(96, dtype=bool))
    grupos = secuencias(planta, plan.inicios)
    assert plan.inicios[0] == 0
    assert [len(g) for g in grupos] == [10, 8]
    assert grupos[1][0] - (grupos[0][-1] + planta.duracion) == planta.cambio_secuencia_cuartos


def test_turnos_respetados():
    planta = Planta(turnos=((6.0, 22.0),), coladas_dia=12)
    precios = dia_sintetico()
    tabla, planes = comparar_dia(planta, precios, perfil_de(precios))
    horas = precios.index.hour
    for plan in planes.values():
        activo = potencia(planta, plan.inicios, len(precios)) > 0
        assert ((horas[activo] >= 6) & (horas[activo] < 22)).all()


def test_regla_evita_el_pico_cuando_cabe():
    precios_arr = 50 + np.arange(96) * 0.01
    precios_arr[72:80] = 300  # pico de 18:00 a 20:00
    precios = pd.Series(precios_arr, index=cuartos_del_dia("2026-02-24"))
    planta = Planta()
    disponible = disponibilidad(planta, precios.index)
    for plan in (plan_regla(planta, precios_arr, disponible, fraccion_cara=8 / 96), plan_optimo(planta, precios_arr, disponible)):
        assert potencia(planta, plan.inicios, 96)[72:80].sum() == 0


def test_verificar_detecta_violaciones():
    planta = Planta(coladas_dia=5, secuencia_min=2, secuencia_max=3, cambio_secuencia_cuartos=4)
    disponible = np.ones(96, dtype=bool)
    verificar_plan(planta, disponible, [0, 4, 8, 16, 20])  # 3 + 2 coladas, cambio de artesa de 4 cuartos
    with pytest.raises(ValueError, match="solapadas"):
        verificar_plan(planta, disponible, [0, 2, 8, 16, 20])
    with pytest.raises(ValueError, match="secuencia de 1"):
        verificar_plan(planta, disponible, [0, 4, 8, 16, 40])
    with pytest.raises(ValueError, match="cambio de artesa"):
        verificar_plan(planta, disponible, [0, 4, 8, 14, 18])
    with pytest.raises(ValueError, match="secuencia de 4"):
        verificar_plan(planta, disponible, [0, 4, 8, 12, 30]) 


def test_horario_fijo_no_mira_los_precios_del_dia():
    perfil = dia_sintetico(semilla=5).to_numpy()
    _, planes1 = comparar_dia(Planta(), dia_sintetico(semilla=1), perfil)
    _, planes2 = comparar_dia(Planta(), dia_sintetico(semilla=2), perfil)
    assert planes1["A"].inicios == planes2["A"].inicios
    assert planes1["O"].inicios != planes2["O"].inicios


def test_omie_usa_la_version_corregida(tmp_path, monkeypatch):
    import txanda.precios as precios_mod

    cuerpo = "MARGINALPDBC;\n" + "\n".join(f"2025;10;30;{h};9;{h};" for h in range(1, 25)) + "\n*\n"
    pedidas = []

    class Respuesta:
        def __init__(self, codigo, texto=""):
            self.status_code, self.content = codigo, texto.encode("latin-1")

        def raise_for_status(self):
            pass

    def falso_get(url, timeout):
        version = int(url.rsplit(".", 1)[1])
        pedidas.append(version)
        return Respuesta(200, cuerpo) if version == 3 else Respuesta(404)

    monkeypatch.setattr(precios_mod.requests, "get", falso_get)
    precios = precios_mod.precios_omie("2025-10-30", cache=tmp_path, cache_ree=None)
    assert pedidas == [1, 2, 3, 4]
    assert (tmp_path / "marginalpdbc_20251030.3").exists()
    assert len(precios) == 96 and precios.iloc[0] == 1.0
