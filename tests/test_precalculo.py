import numpy as np
import pandas as pd

from txanda.backtest import previsores_semana, vigilancia_lear
from txanda.precalculo import cargar_previsiones, grabar_semana, guardar_previsiones, previsores_grabados

from test_prevision import matriz_sintetica


def test_las_previsiones_guardadas_son_las_mismas_que_en_vivo(tmp_path):
    M, fechas = matriz_sintetica(dias=420)
    lunes = fechas[fechas.weekday == 0][-2]
    pos = fechas.get_loc(lunes)
    ruta = guardar_previsiones({lunes: grabar_semana(lunes, M, fechas)}, tmp_path / "previsiones.npz")
    previsores, vigilancia = previsores_grabados(lunes, M, fechas, cargar_previsiones(lunes, ruta))
    vivos = previsores_semana(lunes, M, fechas)
    for clave in ("L", "D"):
        for p in range(pos - 1, pos + 6):
            for k in range(2, 7 - (p - pos)):  # lo que pide la semana: días hasta el domingo
                assert np.array_equal(previsores[clave].predecir(p, k), vivos[clave].predecir(p, k))
    en_vivo = vigilancia_lear(lunes, M, fechas, vivos["L"])
    for p in range(pos - 1, pos + 6):
        assert vigilancia.motivo(p) == en_vivo.motivo(p)
        assert vigilancia.alarma(p) == en_vivo.alarma(p)


def test_semana_no_precalculada(tmp_path):
    assert cargar_previsiones(pd.Timestamp("2030-01-07"), tmp_path / "no_existe.npz") is None
