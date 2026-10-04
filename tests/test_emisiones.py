import json

import numpy as np
import pandas as pd

from txanda.emisiones import intensidad_vector, leer_generacion
from txanda.milp import evaluar
from txanda.planta import Planta


def texto_sintetico(fecha="2025-10-26"):
    filas = []
    for h in ["00", "01", "2A", "2B", "03"] + [f"{h:02d}" for h in range(4, 24)]:
        for m in range(0, 60, 5):
            filas.append({"ts": f"{fecha} {h}:{m:02d}", "dem": 30000, "car": 0, "cc": 4000 if h in ("2A", "2B") else 2000,
                          "gf": 0, "cogenResto": 1000, "vap": 200})
    filas.append({"ts": "2025-10-27 00:00", "dem": 1, "car": 0, "cc": 99999, "gf": 0, "cogenResto": 0, "vap": 0})
    return "x(" + json.dumps({"valoresHorariosGeneracion": filas}) + ");"


def test_leer_generacion_une_las_horas_repetidas_y_descarta_otros_dias():
    gen = leer_generacion(texto_sintetico(), "2025-10-26")
    assert len(gen) == 96
    assert gen.loc["02:00", "cc"] == 4000 and gen.loc["05:00", "cc"] == 2000
    assert gen["cc"].max() < 99999


def test_intensidad_reproduce_el_total_oficial():
    gen = leer_generacion(texto_sintetico(), "2025-10-26")
    oficiales = pd.Series({"Ciclo combinado": 18000.0, "Cogeneración": 9000.0, "Residuos no renovables": 500.0, "Carbón": 0.0})
    intensidad = intensidad_vector(gen, oficiales)
    assert np.isclose((intensidad * gen["dem"].to_numpy() * 0.25).sum(), oficiales.sum())
    assert intensidad[8] > intensidad[40]  # a las 02:00 hubo más ciclo combinado


def test_evaluar_suma_emisiones():
    planta = Planta(perfil_mw=(10.0, 10.0), coladas_dia=1, secuencia_min=1, secuencia_max=2)
    r = evaluar(planta, np.zeros(8), [0], intensidad=np.full(8, 0.1))
    assert np.isclose(r["emisiones_t"], 0.5) and np.isclose(r["kgco2_por_t"], 5.0)
