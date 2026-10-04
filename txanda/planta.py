"""Parámetros de la planta tipo.

Los valores por defecto describen una acería de horno de arco eléctrico (EAF)
estilizada: órdenes de magnitud habituales en la literatura, no datos de una
planta concreta. Hay que calibrarlos y hacerles análisis de sensibilidad.
"""
from __future__ import annotations

from dataclasses import dataclass

DT_H = 0.25  # duración de un periodo del mercado diario (15 min), en horas


@dataclass(frozen=True)
class Planta:
    # Potencia media del horno en cada cuarto de hora de una colada, en MW. Su
    # longitud es la duración colada a colada (tap-to-tap): 4 cuartos = 60 min.
    # Carga y fusión concentran la potencia; afino y colada consumen poco.
    perfil_mw: tuple[float, ...] = (60.0, 64.0, 28.0, 8.0)  # 40 MWh por colada
    toneladas_colada: float = 100.0  # 40 MWh / 100 t ≈ 400 kWh/t
    coladas_dia: int = 18
    # Flexibilidad semanal: el tonelaje se compromete por semana (7 × coladas_dia) y
    # cada día puede producir entre coladas_dia − holgura_menos y coladas_dia + holgura_mas.
    holgura_menos: int = 6
    holgura_mas: int = 4

    # La colada continua exige que las coladas de una secuencia vayan seguidas,
    # sin huecos. Cada secuencia empieza con una artesa (tundish) nueva.
    secuencia_min: int = 4  # coladas mínimas por secuencia
    secuencia_max: int = 10  # vida de la artesa, en coladas
    cambio_secuencia_cuartos: int = 4  # parada mínima entre secuencias (cambio de artesa)
    coste_arranque_eur: float = 2500.0  # artesa, pérdidas térmicas, rendimiento (supuesto)

    # Franjas con personal, en horas locales [inicio, fin). Por defecto, tres turnos.
    turnos: tuple[tuple[float, float], ...] = ((0.0, 24.0),)

    def __post_init__(self):
        if not self.perfil_mw or min(self.perfil_mw) < 0:
            raise ValueError("perfil_mw debe tener al menos un cuarto y potencias no negativas")
        if not 1 <= self.secuencia_min <= self.secuencia_max:
            raise ValueError("Hace falta 1 <= secuencia_min <= secuencia_max")
        if self.holgura_menos < 0 or self.holgura_mas < 0:
            raise ValueError("Las holguras diarias no pueden ser negativas")
        if self.coladas_dia < 1 or self.cambio_secuencia_cuartos < 1:
            raise ValueError("coladas_dia >= 1 y cambio_secuencia_cuartos >= 1")

    @property
    def duracion(self) -> int:
        """Cuartos de hora que ocupa una colada."""
        return len(self.perfil_mw)

    @property
    def coladas_dia_min(self) -> int:
        return max(0, self.coladas_dia - self.holgura_menos)

    @property
    def coladas_dia_max(self) -> int:
        return self.coladas_dia + self.holgura_mas

    @property
    def energia_colada_mwh(self) -> float:
        return DT_H * sum(self.perfil_mw)
