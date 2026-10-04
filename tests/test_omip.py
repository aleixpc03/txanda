import numpy as np
import pandas as pd

from txanda.omip import leer_sesion
from txanda.prevision import PrevisorFuturos, ingenua

HTML = """<html><body><table>
<tr><td>Contract name</td><td>Price</td><td>D (€/MWh)</td><td>D-1 (€/MWh)</td></tr>
<tr><td>ISIN Code: X1 FTB D Mo23Feb-26</td><td>n.a.</td><td>36.05</td><td>n.a.</td></tr>
<tr><td>ISIN Code: X2 FTB D Tu24Feb-26</td><td>n.a.</td><td>n.a.</td><td>n.a.</td></tr>
</table>
<table><tr><td>Contract name</td><td>D (€/MWh)</td></tr><tr><td>ISIN Code: X3 FTB WE 28Feb-26</td><td>29.89</td></tr></table>
</body></html>"""


def test_leer_sesion_solo_contratos_diarios():
    sesion = leer_sesion(HTML)
    assert sesion.entrega.astype(str).tolist() == ["2026-02-23", "2026-02-24"]
    assert sesion.precio.iloc[0] == 36.05 and np.isnan(sesion.precio.iloc[1])
    assert len(leer_sesion("<html>sin sesión</html>")) == 0


def test_futuros_usan_la_ultima_sesion_anterior_a_la_decision():
    fechas = pd.date_range("2026-02-01", periods=40)
    M = np.tile(np.linspace(10, 50, 96), (40, 1)) + np.arange(40)[:, None]
    futuros = pd.DataFrame({
        "sesion": pd.to_datetime(["2026-02-19", "2026-02-20", "2026-02-23"]).date,
        "entrega": pd.to_datetime(["2026-02-25", "2026-02-25", "2026-02-25"]).date,
        "precio": [40.0, 45.0, 99.0],
    })
    previsor = PrevisorFuturos(M, fechas, futuros)
    pos_D = fechas.get_loc(pd.Timestamp("2026-02-23"))  # decide el lunes: vale la sesión del viernes 20
    prevision = previsor.predecir(pos_D, 2)
    assert prevision.mean() == 45.0
    forma = M[pos_D + 2 - 7]
    assert np.allclose(prevision - prevision.mean(), forma - forma.mean())
    assert np.array_equal(previsor.predecir(pos_D, 3), ingenua(M, pos_D, 3))  # sin cotización: ingenua


def test_hibrido_elige_previsor_segun_antelacion():
    from txanda.prevision import PrevisorHibrido

    class Fijo:
        def __init__(self, valor):
            self.valor = valor

        def predecir(self, pos_D, k):
            return np.full(96, self.valor)

    hibrido = PrevisorHibrido(Fijo(1.0), Fijo(2.0), desde_k=3)
    assert hibrido.predecir(10, 2)[0] == 1.0
    assert all(hibrido.predecir(10, k)[0] == 2.0 for k in range(3, 8))
