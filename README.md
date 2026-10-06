# Txanda

Planificación de coladas de una acería de horno eléctrico frente al precio cuartohorario
de la electricidad. Proyecto para Donostia Meeting Minds 2026, Categoría Empresa.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Uso

```bash
.venv/bin/python -m txanda dia 2026-02-23                  # un día: tabla y gráfico en salidas/
.venv/bin/python -m txanda periodo 2025-10-01 2026-09-29   # backtest día a día
.venv/bin/python -m txanda semana 2026-02-23               # una semana (lunes): tabla, errores de previsión y gráfico
.venv/bin/python -m txanda semanas 2025-10-06 2026-09-21   # backtest de todas las semanas (≈ 8 min con 8 núcleos)
.venv/bin/python -m txanda semanas 2025-10-06 2026-09-21 --coste-arranque 20000 --secuencia-min 6 --rango 16-20 --cambio-artesa 0.25 --gap 0.0001 --etiqueta industrial   # otra planta, resultados en salidas/escenarios/
.venv/bin/python -m txanda sensibilidad-arranque 2025-10-06 2026-09-21      # costes de arranque 0–10.000 €
.venv/bin/python -m txanda sensibilidad-flexibilidad 2025-10-06 2026-09-21  # rangos diarios 18–18 a 0–22
.venv/bin/python -m pytest -q                              # tests
```

## Memoria

`memoria/Memoria_Txanda_borrador.docx` (y su PDF) es el borrador de la memoria del concurso: 8
páginas con los apartados que piden las bases. Lo marcado en amarillo lo tiene que completar
el equipo (nombres, captura del prototipo, inversión).

## Prototipo

```bash
.venv/bin/streamlit run app.py
```

Abre `http://localhost:8501`. En **Planificar una semana** se elige la semana (octubre de
2025 a septiembre de 2026), la previsión, las coladas por día, el rango diario, el coste
de arranque y los turnos. La app simula cómo habría decidido Txanda día a día y muestra el
precio con la previsión del domingo, la potencia del horno con Txanda, el horario fijo y
el oráculo, la comparación de estrategias y la orden de fabricación descargable en CSV.
**Resultados del año** muestra el backtest y los gráficos de sensibilidad si ya existen
en `salidas/`. Una semana tarda unos 30 s la primera vez; después queda en caché.

La app sigue a la memoria: cuatro indicadores, la comparación con todas las estrategias del
benchmark (siete; nueve si se activan los futuros de OMIP con `TXANDA_OMIP=1`) y la tabla del
año como la tabla 5. Las emisiones están calculadas pero ocultas (`MOSTRAR_CO2 = False` en
`app.py`) porque la memoria las deja como trabajo pendiente.

Con Docker (montando las carpetas de datos y resultados):

```bash
docker build -t txanda .
docker run -p 8501:8501 -v "$PWD/datos:/app/datos" -v "$PWD/salidas:/app/salidas" txanda
```

## Publicar en Streamlit Community Cloud

En la nube los precios se descargan de Red Eléctrica, un mes por petición: la primera carga
tarda alrededor de un minuto y después queda en caché.

1. Subir el repositorio a GitHub como privado (desde esta carpeta):
   `gh repo create txanda --private --source . --push`
2. Entrar en [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub, crear una
   app nueva y elegir el repositorio `txanda`, la rama `main` y el fichero `app.py`. En la
   configuración avanzada, Python 3.12.
3. Si el repositorio es privado, la app también lo es: solo la ven las personas invitadas por
   correo desde *Settings → Sharing*.
4. Cada vez que se repitan los backtests en local, actualizar la versión publicable y subirla:
   `.venv/bin/python -m txanda publicar`, y después `git add resultados && git commit` y `git push`.

Los precios se descargan de OMIE (fichero `marginalpdbc`, columna de España, última versión
publicada) y se guardan en `datos/omie/`. Si un día no está descargado, se toma antes de Red
Eléctrica, que publica la misma serie (comprobado cuarto a cuarto) y sirve un mes por petición. Antes del 1-10-2025 el mercado era horario y cada
precio se repite en sus cuatro cuartos. Los días de cambio de hora tienen 92 o 100 cuartos.

## Estrategias

Todas producen las mismas coladas a la semana y se evalúan igual: energía a precio real
más el coste de cada arranque de secuencia. Cada día D, a las 12:00, se conocen los precios
de D+1; las estrategias que deciden día a día fijan las coladas de D+1 y vuelven a resolver
al día siguiente (horizonte rodante).

| | Estrategia | Qué sabe | Cupos |
|---|---|---|---|
| A | Horario fijo | Perfil medio de precios del mes anterior; se repite todo el mes | 18 coladas/día |
| B | Paradas en horas caras | Qué cuartos de D+1 son caros (el 25 % más caro) | 18 coladas/día |
| OD | Óptimo diario | Precio exacto de D+1 | 18 coladas/día |
| C | Semanal, previsión ingenua | D+1 exacto; D+2…domingo = mismo día de la semana anterior | 126/semana, 12–22/día |
| L | Semanal, previsión LEAR | D+1 exacto; D+2…domingo = LASSO autorregresivo (Lago et al., 2021) | 126/semana, 12–22/día |
| D | Semanal, previsión ML | D+1 exacto; D+2…domingo = previsión con árboles de gradiente | 126/semana, 12–22/día |
| O | Oráculo semanal | Toda la semana de antemano. No es implementable: es el techo | 126/semana, 12–22/día |

Las previsiones están en `txanda/prevision.py` y se reentrenan cada semana con el último año
(5–6 s cada una en un portátil). `tests/test_prevision.py` comprueba que no usan datos del futuro.

- **LEAR**: un LASSO por horizonte y cuarto de hora, con λ elegido por AIC, sobre los precios
  horarios de D+1, D, D−1 y del mismo día de la semana anterior, con la transformación asinh
  estandarizada de la literatura. Es la referencia estándar en previsión de precios eléctricos.
- **ML**: árboles con gradiente (HistGradientBoosting) sobre precios recientes y calendario.
- **Futuros de OMIP: no se usan.** Sus condiciones no permiten usar sus datos en el proyecto.
  El código (`txanda/omip.py`, estrategias F y H) se conserva, pero solo se activa con
  `TXANDA_OMIP=1`; ni la app ni los backtests lo usan por defecto.

## Formulación

Formulación por secuencias (`txanda/milp.py`). `t` es el cuarto de hora del horizonte (un
día o una semana), `d` la duración de una colada en cuartos, `G` el cambio de artesa,
`Lmin…Lmax` las coladas por secuencia y `c[s]` el coste de una colada que empieza en `s`.

```
y[s,L] ∈ {0,1}   empieza en s una secuencia de L coladas seguidas; ocupa el horno en
                 [s, s + L·d) y deja el cambio de artesa en [s + L·d, s + L·d + G)

min  Σ y[s,L]·( Σ_{j<L} c[s + j·d] + w )

s.a. un camino de 0 a T: en cada cuarto t el horno pasa a t+1 parado o empieza una
     secuencia y salta a s + L·d + G                         (sin solapes, cambio de artesa)
     mín_k ≤ Σ y[s,L]·n_k(s,L) ≤ máx_k            ∀ cupo k    (tonelaje por día y por semana)
     y[s,L] = 0 si alguna colada no cabe o cae fuera de turno
     las secuencias ya comprometidas se mantienen; si siguen abiertas, solo se elige
     cuánto se alargan
```

Con `p_k` la potencia de la colada en su cuarto `k` y `Δt = 0,25 h`,
`c[s] = Σ_k Δt·p_k·precio[s+k]`. Sin los cupos es un camino mínimo en un grafo acíclico,
cuya relajación lineal es entera. Una semana se resuelve en 0,5–2 s; la primera formulación
(una binaria por colada) daba los mismos óptimos en unos 50 s.

`tests/test_milp_dia.py` comprueba contra fuerza bruta que el MILP encuentra el óptimo, y
cada plan se valida con `verificar_plan`, que no usa el MILP.

## Resultados: 51 semanas, del 6-10-2025 al 27-9-2026

| | Coste | €/t | Ahorro frente a A | Parte del ahorro máximo |
|---|---|---|---|---|
| A · Horario fijo | 17,14 M€ | 26,7 | — | 0 % |
| B · Paradas en horas caras | 17,36 M€ | 27,0 | −1,2 % | −33 % |
| OD · Óptimo diario | 16,99 M€ | 26,4 | 0,9 % | 24 % |
| C · Semanal, previsión ingenua | 16,99 M€ | 26,4 | 0,9 % | 24 % |
| L · Semanal, previsión LEAR | 16,82 M€ | 26,2 | 1,9 % | 50 % |
| D · Semanal, previsión ML | 16,97 M€ | 26,4 | 1,0 % | 27 % |
| O · Oráculo semanal | 16,50 M€ | 25,7 | 3,8 % | 100 % |

Comparaciones semana a semana (prueba de Wilcoxon pareada):

- L gana a la previsión ingenua (33 de 51 semanas, p = 0,025) y al óptimo diario (37
  semanas, p = 0,002).
- L gana a la previsión ML (32 semanas, p = 0,013). D no se distingue de C ni de OD.
- Todas las estrategias optimizadas ganan a la práctica del sector B; OD gana 49 de 51
  semanas (p = 6·10⁻¹⁰). B pierde frente al horario fijo A (p = 0,004).
- Error medio de la previsión (€/MWh): la ingenua, 24–28; LEAR es la mejor a dos días
  (16) y a cinco o más se acerca a la ingenua. Una previsión más precisa no garantiza un
  plan más barato: la ML acierta más que la ingenua y planifica casi igual.
- **Emisiones** (`txanda/emisiones.py`): con la intensidad de CO₂ del sistema peninsular en cada
  cuarto de hora (generación cada 5 minutos y emisiones oficiales diarias de Red Eléctrica; la
  serie reconstruida reproduce el total diario oficial), el horario fijo se asocia a 33,6 kg
  CO₂/t. L lo reduce un 1,7 % (≈ 370 t al año en la planta tipo), C un 1,6 %, D un 1,4 % y
  el oráculo un 2,6 %. El óptimo diario no cambia las emisiones (−0,0 %) y B las sube un 0,2 %:
  la reducción viene de repartir la producción entre días, no de moverla dentro del día.

## Sensibilidad

**Coste de arranque** (0 a 10.000 €, `resultados/sensibilidad_arranque.png`). La clasificación
no cambia: L captura entre el 50 % y el 64 % del ahorro máximo y gana al óptimo diario con
cualquier coste (p ≤ 0,003). A la previsión ingenua le gana con significación a partir de
2.500 €. B queda siempre por debajo del horario fijo. El techo del
oráculo se mueve poco (3,8–4,3 %). Una artesa dura como mucho 10 coladas, así que una semana
de 126 necesita al menos 13 secuencias: los arranques solo varían entre 13 y 16 por semana.

**Flexibilidad diaria** (rango de coladas por día con 126 a la semana,
`resultados/sensibilidad_flexibilidad.png`):

| Rango diario | Techo O | L | C (ingenua) | D (ML) |
|---|---|---|---|---|
| 18–18 (sin flexibilidad) | 1,8 % | 1,5 % | 1,4 % | 1,4 % |
| 16–20 | 3,1 % | 1,9 % | 1,4 % | 1,5 % |
| 12–22 (caso base) | 3,8 % | 1,9 % | 0,9 % | 1,0 % |
| 0–22 | 4,0 % | 2,0 % | 0,5 % | 0,8 % |

Ahorro frente al horario fijo A. Con ±2 coladas al día (16–20) L ya consigue casi todo su
ahorro (≈ 333 k€/año en la planta tipo); más flexibilidad sube el techo pero no lo que
capturan, porque el límite es la previsión. Con previsiones malas, la flexibilidad perjudica:
C y D ahorran menos cuanto más margen tienen, porque desplazan producción hacia días que
creían baratos y no lo eran. Sin flexibilidad (18–18), mirar la semana entera sigue ganando
al óptimo diario (1,5 % frente a 0,9 %) por cómo se enlazan las secuencias de un día con el
siguiente.

## Escenario industrial

Con arranques caros y menos margen (w = 20.000 €, secuencias de al menos 6 coladas, 16–20
coladas al día y 0,25 h de cambio de artesa), planificar la semana vale más: LEAR ahorra un
3,3 % frente al horario fijo (≈ 984 k€/año, de los que 285 son arranques evitados) y captura
alrededor del 80 % del ahorro máximo; gana al óptimo diario en 50 de 51 semanas y a la
previsión ingenua en 32 (p = 0,016). El óptimo diario apenas mejora al horario fijo (0,2 %)
porque lo que ahorra en energía lo pierde en arranques. Con estos parámetros el solver es
más lento: se usó `--gap 0.0001` y en 42 de 357 problemas se agotó el tiempo y se tomó la
mejor solución encontrada (el resumen lo indica).

## Supuestos de la planta tipo (`txanda/planta.py`)

| Parámetro | Valor | Nota |
|---|---|---|
| Perfil de colada | 60 · 64 · 28 · 8 MW por cuarto | 60 min colada a colada, 40 MWh |
| Tamaño de colada | 100 t | ≈ 400 kWh/t |
| Coladas | 18/día, 126/semana | entre 12 y 22 al día en las estrategias semanales |
| Secuencia | 4 a 10 coladas | vida de la artesa |
| Cambio de artesa | 1 h | |
| Coste de arranque | 2.500 € | supuesto: falta calibrarlo |
| Turnos | 24 h | |

Son órdenes de magnitud de la literatura, no datos de una planta concreta.

## Pendiente

1. Previsiones de eólica, solar y demanda (ESIOS), precio del gas (MIBGAS) y festivos.
2. Estrategia E: entrenar la previsión por el coste de la decisión (SPO+, PyEPO).
3. Peajes y cargos por periodo tarifario (P1–P6) y otras cargas (horno cuchara, laminación).
