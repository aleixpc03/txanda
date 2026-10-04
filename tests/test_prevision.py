import numpy as np
import pandas as pd
import pytest

from txanda.prevision import PrevisorLEAR, PrevisorML, ingenua, rasgos


def matriz_sintetica(dias=120, semilla=0):
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2025-01-01", periods=dias)
    q = np.arange(96)
    base = 60 - 40 * np.exp(-((q - 54) ** 2) / 200) + 40 * np.exp(-((q - 82) ** 2) / 40)
    finde = np.where(fechas.weekday >= 5, -15.0, 0.0)
    M = base[None, :] + finde[:, None] + rng.normal(0, 5, (dias, 96))
    return M, fechas


def test_ingenua_es_la_semana_anterior():
    M, _ = matriz_sintetica()
    assert np.array_equal(ingenua(M, 50, 3), M[46])


@pytest.mark.parametrize("clase", [PrevisorML, PrevisorLEAR])
def test_prevision_no_usa_el_futuro(clase):
    M, fechas = matriz_sintetica(dias=200)
    previsor = clase(M, fechas).entrenar(ultimo_conocido=150)
    pos_D = 160
    antes = [previsor.predecir(pos_D, k) for k in range(2, 8)]
    alterada = M.copy()
    alterada[pos_D + 2 :] += 1000.0  # todo lo que aún no se conoce en D
    previsor.M = alterada
    despues = [previsor.predecir(pos_D, k) for k in range(2, 8)]
    for a, b in zip(antes, despues):
        assert np.allclose(a, b)
    for k in range(2, 8):
        assert rasgos(M, fechas, pos_D, k).shape == (96, 13)


@pytest.mark.parametrize("clase", [PrevisorML, PrevisorLEAR])
def test_aprende_el_patron_semanal(clase):
    M, fechas = matriz_sintetica(dias=200)
    previsor = clase(M, fechas).entrenar(ultimo_conocido=150)
    errores_ml, errores_ing = [], []
    for pos_D in range(150, 190):
        for k in range(2, 8):
            errores_ml.append(np.abs(previsor.predecir(pos_D, k) - M[pos_D + k]).mean())
            errores_ing.append(np.abs(ingenua(M, pos_D, k) - M[pos_D + k]).mean())
    assert np.mean(errores_ml) < np.mean(errores_ing)
