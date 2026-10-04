"""Previsión de precios de D+2 a D+7, hecha el día D a las 12:00, cuando ya se conoce D+1.

Trabaja sobre la matriz diaria M (una fila por día, 96 cuartos por hora de reloj):
`pos_D` es la fila del día en que se decide y `k` la distancia al día previsto.
Ninguna previsión usa filas posteriores a pos_D + 1.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC

HORIZONTES = range(2, 8)
HISTORIA_MIN = 14  # días previos que necesitan los rasgos


def ingenua(M: np.ndarray, pos_D: int, k: int) -> np.ndarray:
    """El mismo cuarto de hora del mismo día de la semana anterior."""
    return M[pos_D + k - 7]


def rasgos(M: np.ndarray, fechas: pd.DatetimeIndex, pos_D: int, k: int) -> np.ndarray:
    """Una fila por cuarto de hora del día previsto, solo con información conocida en D."""
    t = pos_D + k
    ultimo, hoy, semana, semana2 = M[pos_D + 1], M[pos_D], M[t - 7], M[t - 14]
    media7 = M[pos_D - 5 : pos_D + 2].mean(axis=0)
    fecha = fechas[t]
    constantes = [k, fecha.weekday(), fecha.month, ultimo.mean(), ultimo.max(), ultimo.min(), semana.mean()]
    return np.column_stack(
        [np.arange(96), ultimo, hoy, semana, semana2, media7] + [np.full(96, float(v)) for v in constantes]
    )


class PrevisorML:
    """Árboles con gradiente (HistGradientBoosting) sobre precios recientes y calendario."""

    def __init__(self, M: np.ndarray, fechas: pd.DatetimeIndex, ventana_dias: int = 365, semilla: int = 0):
        self.M, self.fechas, self.ventana, self.semilla = M, fechas, ventana_dias, semilla
        self.modelo = None

    def entrenar(self, ultimo_conocido: int) -> "PrevisorML":
        """Entrena con las previsiones pasadas cuyo día objetivo ya se conoce."""
        X, y = [], []
        for pos_D in range(max(HISTORIA_MIN, ultimo_conocido - self.ventana), ultimo_conocido - 1):
            for k in HORIZONTES:
                if pos_D + k > ultimo_conocido:
                    break
                X.append(rasgos(self.M, self.fechas, pos_D, k))
                y.append(self.M[pos_D + k])
        self.modelo = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, l2_regularization=1.0,
            categorical_features=[7], random_state=self.semilla,  # columna 7: día de la semana
        ).fit(np.vstack(X), np.concatenate(y))
        return self

    def predecir(self, pos_D: int, k: int) -> np.ndarray:
        return self.modelo.predict(rasgos(self.M, self.fechas, pos_D, k))


class PrevisorLEAR:
    """LEAR (Lago et al., 2021, epftoolbox) adaptado a D+2…D+7.

    Un LASSO por horizonte k y cuarto de hora, con λ elegido por AIC. Entradas: precios
    horarios de D+1, D, D−1 y del mismo día de la semana anterior, más el día de la
    semana. Precios y entradas pasan por la transformación asinh estandarizada de la
    literatura, que doma los picos y admite precios negativos.
    """

    def __init__(self, M: np.ndarray, fechas: pd.DatetimeIndex, ventana_dias: int = 364):
        self.M, self.fechas, self.ventana = M, fechas, ventana_dias
        self.pesos: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _z(self, p):
        return np.arcsinh((p - self.mediana) / self.mad)

    def _rasgos(self, pos_D: int, k: int) -> np.ndarray:
        t = pos_D + k
        semanal = t - 7 if t - 7 < pos_D - 1 else t - 14
        horas = [self.M[fila].reshape(24, 4).mean(axis=1) for fila in (pos_D + 1, pos_D, pos_D - 1, semanal)]
        dia = np.eye(7)[self.fechas[t].weekday()]
        return np.concatenate([self._z(np.concatenate(horas)), dia])

    def entrenar(self, ultimo_conocido: int) -> "PrevisorLEAR":
        inicio = max(HISTORIA_MIN, ultimo_conocido - self.ventana)
        conocidos = self.M[inicio : ultimo_conocido + 1]
        self.mediana = np.median(conocidos)
        self.mad = max(np.median(np.abs(conocidos - self.mediana)) / 0.6745, 1e-6)
        for k in HORIZONTES:
            posiciones = range(inicio, ultimo_conocido - k + 1)
            X = np.vstack([self._rasgos(p, k) for p in posiciones])
            if len(X) <= X.shape[1] + 1:
                raise ValueError(f"LEAR necesita más de {X.shape[1] + 1} días de historia; hay {len(X)}")
            Y = self._z(np.vstack([self.M[p + k] for p in posiciones]))
            media, escala = X.mean(axis=0), X.std(axis=0) + 1e-9
            Xs = (X - media) / escala
            W, b = np.zeros((96, X.shape[1])), np.zeros(96)
            for q in range(96):
                with warnings.catch_warnings():  # LARS avisa al cortar el camino; no afecta al ajuste
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    lasso = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xs, Y[:, q])
                W[q], b[q] = lasso.coef_ / escala, lasso.intercept_ - (lasso.coef_ / escala) @ media
            self.pesos[k] = (W, b)
        return self

    def predecir(self, pos_D: int, k: int) -> np.ndarray:
        W, b = self.pesos[k]
        return self.mediana + self.mad * np.sinh(W @ self._rasgos(pos_D, k) + b)


class PrevisorFuturos:
    """Nivel diario del mercado de futuros (OMIP, carga base) con la forma del mismo día de
    la semana anterior.

    Para decidir el día D usa la última sesión cerrada antes de D: OMIP publica los
    precios de referencia al cierre, por la tarde. Si ninguna sesión cotizó ese día de
    entrega, devuelve la previsión ingenua.
    """

    def __init__(self, M: np.ndarray, fechas: pd.DatetimeIndex, futuros: pd.DataFrame):
        self.M, self.fechas = M, fechas
        self.cotizaciones = {}
        for entrega, grupo in futuros.dropna(subset=["precio"]).groupby("entrega"):
            grupo = grupo.sort_values("sesion")
            self.cotizaciones[pd.Timestamp(entrega).date()] = (
                np.array(grupo.sesion.tolist(), dtype="datetime64[D]"), grupo.precio.to_numpy(dtype=float)
            )

    def entrenar(self, ultimo_conocido: int) -> "PrevisorFuturos":
        return self  # no hay nada que ajustar: la previsión es la del mercado

    def nivel(self, dia_decision, entrega) -> float | None:
        """Precio de referencia de `entrega` en la última sesión anterior a `dia_decision`."""
        datos = self.cotizaciones.get(pd.Timestamp(entrega).date())
        if datos is None:
            return None
        sesiones, precios = datos
        i = np.searchsorted(sesiones, np.datetime64(pd.Timestamp(dia_decision).date(), "D")) - 1
        return float(precios[i]) if i >= 0 else None

    def predecir(self, pos_D: int, k: int) -> np.ndarray:
        t = pos_D + k
        forma = self.M[t - 7]
        nivel = self.nivel(self.fechas[pos_D], self.fechas[t])
        return forma if nivel is None else forma - forma.mean() + nivel


class PrevisorHibrido:
    """Combina dos previsores según la antelación: `cerca` hasta k < desde_k, `lejos` a partir de ahí.

    Con LEAR cerca y futuros de OMIP lejos aprovecha a cada uno donde acierta más.
    """

    def __init__(self, cerca, lejos, desde_k: int = 3):
        self.cerca, self.lejos, self.desde_k = cerca, lejos, desde_k

    def entrenar(self, ultimo_conocido: int) -> "PrevisorHibrido":
        return self  # usa los previsores ya entrenados

    def predecir(self, pos_D: int, k: int) -> np.ndarray:
        return (self.cerca if k < self.desde_k else self.lejos).predecir(pos_D, k)
