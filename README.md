# Laboratorio Banco Colombia - Simulación M/M/1

Este repositorio implementa una simulación estocástica del laboratorio bancario con **600 corridas** y horizonte de **8 horas (480 minutos)** por réplica.

## Qué resuelve

El script `bank_simulation.py` cubre los 5 puntos solicitados:

1. **Cajero con menor y mayor tiempo promedio de atención** (por escenario).
2. **Promedio de usuarios por tipo** (rápido, normal, lento, muy lento).
3. **Total de usuarios por tipo por réplica** y réplica con menor conteo por tipo.
4. **Necesidad de nuevo cajero** comparando 3 vs 4 cajas.
5. **Configuración óptima** por menor tiempo promedio de espera.

Además incluye:

- Validación de estabilidad del sistema por escenario (**ρ < 1**).
- Métricas M/M/1 por cajero: **L, Lq, W, Wq, utilización**.
- Intervalo de confianza al 95% para la espera promedio.
- Evaluación de escenarios:
  - `3_cajas_mixtas`
  - `1_retiro_2_pagos`
  - `2_retiros_1_pago`
  - `4_cajas_3_retiros_1_pago`

## Ejecución

```bash
python /tmp/workspace/Darkcode-Git/Laboratorio-Banco-Colombia/bank_simulation.py
```

Salida principal:

- Configuración óptima recomendada.
- Recomendación sobre nuevo cajero.
- Resumen de esperas promedio e IC95 por escenario.

Si `matplotlib` está disponible, también genera `comparacion_esperas.png`.

## Pruebas

```bash
python -m unittest discover -v
```

Las pruebas validan:

- Probabilidades del modelo.
- Estabilidad de los escenarios.
- Estructura mínima de resultados para análisis del laboratorio.
