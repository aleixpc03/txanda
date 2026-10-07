import numpy as np

from txanda.planta import Planta
from txanda.prevision import Vigilancia
from txanda.semana import reparto, simular_semana
from tests.test_semana import datos_sinteticos

POS_LUNES = 21  # fila del lunes en la matriz de datos_sinteticos


class Perfecta:
    """Acierta siempre: devuelve el precio real del día previsto."""

    def __init__(self, M):
        self.M = M

    def predecir(self, pos_D, k):
        return self.M[pos_D + k]


class Desviada(Perfecta):
    """Se equivoca en 500 €/MWh en cada cuarto: siempre peor que la ingenua."""

    def predecir(self, pos_D, k):
        return self.M[pos_D + k] + 500.0


class Rota:
    def predecir(self, pos_D, k):
        raise RuntimeError("modelo sin entrenar")


def simular(previsor, vigilancia):
    semana, M, fechas = datos_sinteticos()
    tabla, planes, _ = simular_semana(Planta(), semana, M, fechas, {"L": previsor}, perfil=M[:7].mean(axis=0),
                                      vigilancias={"L": vigilancia})  # verifica cada plan
    return semana, tabla, planes


def coladas_por_dia(semana, plan):
    lim = semana.limites
    return [sum(lim[i] <= s < lim[i + 1] for s in plan.inicios) for i in range(7)]


def test_reparto_suma_lo_que_queda_a_partes_iguales():
    assert reparto(126, 7) == [18] * 7
    assert reparto(100, 6) == [17, 17, 17, 17, 16, 16]
    assert sum(reparto(83, 5)) == 83


def test_vigilancia_explica_por_que_no_se_fia():
    _, M, _ = datos_sinteticos()
    pos_D, horizontes = POS_LUNES + 2, range(2, 5)  # el miércoles se prevé hasta el domingo
    assert Vigilancia(M, [(0, Perfecta(M))]).motivo(pos_D, horizontes) is None
    assert Vigilancia(M, [(0, Perfecta(M))]).alarma(pos_D) is None
    mala = Vigilancia(M, [(0, Desviada(M))])
    assert mala.motivo(pos_D, horizontes) is None  # acertar poco solo avisa…
    assert "ingenua" in mala.alarma(pos_D)
    assert "ingenua" in Vigilancia(M, [(0, Desviada(M))], por_acierto=True).motivo(pos_D, horizontes)  # …salvo que se pida
    assert "no se puede calcular" in Vigilancia(M, [(0, Rota())]).motivo(pos_D, horizontes)
    assert "ninguna previsión" in Vigilancia(M, [(pos_D + 1, Perfecta(M))]).motivo(pos_D, horizontes)
    con_hueco = M.copy()
    con_hueco[pos_D] = np.nan
    assert "faltan precios" in Vigilancia(con_hueco, [(0, Perfecta(M))]).motivo(pos_D, horizontes)


def test_con_una_prevision_fiable_no_vuelve_nunca_y_coincide_con_L():
    _, M, _ = datos_sinteticos()
    _, tabla, planes = simular(Perfecta(M), Vigilancia(M, [(0, Perfecta(M))]))
    assert planes["LR"].respaldo == ()
    assert planes["LR"].inicios == planes["L"].inicios
    assert tabla.loc["LR", "dias_respaldo"] == 0


def test_una_prevision_mala_solo_avisa_y_LR_coincide_con_L():
    _, M, _ = datos_sinteticos()
    _, _, planes = simular(Desviada(M), Vigilancia(M, [(0, Desviada(M))]))
    assert planes["LR"].respaldo == ()
    assert planes["LR"].inicios == planes["L"].inicios


def test_vigilando_el_acierto_una_prevision_mala_vuelve_cada_dia_al_optimo_diario():
    _, M, _ = datos_sinteticos()
    _, tabla, planes = simular(Desviada(M), Vigilancia(M, [(0, Desviada(M))], por_acierto=True))
    assert [j for j, _ in planes["LR"].respaldo] == list(range(7))
    assert planes["LR"].inicios == planes["OD"].inicios
    assert tabla.loc["LR", "dias_respaldo"] == 7


def test_si_faltan_datos_a_media_semana_reparte_lo_que_queda_y_cumple_la_semana():
    _, M, _ = datos_sinteticos()
    con_hueco = M.copy()
    con_hueco[POS_LUNES + 2] = np.nan  # el miércoles no llegan los precios a la vigilancia
    semana, tabla, planes = simular(Perfecta(M), Vigilancia(con_hueco, [(0, Perfecta(M))]))
    respaldo = planes["LR"].respaldo
    assert [j for j, _ in respaldo] == [2, 3, 4, 5, 6]
    assert all("faltan precios" in motivo for _, motivo in respaldo)
    por_dia = coladas_por_dia(semana, planes["LR"])
    assert sum(por_dia) == 7 * Planta().coladas_dia
    assert por_dia[2:] == reparto(7 * Planta().coladas_dia - sum(por_dia[:2]), 5)
