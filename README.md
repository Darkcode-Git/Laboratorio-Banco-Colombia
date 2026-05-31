# Laboratorio-Banco-Colombia

Laboratorio: Problema Bancario. Simulación de sistemas de colas M/M/1 para el análisis
y optimización de cajeros del Banco de Colombia.

## Simulación DES (M/M/1)

El script `simulacion_banco.py` implementa simulación de eventos discretos (DES) con:
- Validación de estabilidad (ρ < 1) en cada réplica.
- Cálculo de Wq, Lq y utilización.
- Intervalos de confianza al 95% con t-student.
- Gráficas de promedio acumulado para identificar warm-up.

### Requisitos

```bash
pip install -r requirements.txt
```

### Ejecución

Las tasas se expresan en **clientes por hora**. El tiempo se simula en minutos.

```bash
python simulacion_banco.py \
  --arrival-rate 60 \
  --service-rate-withdrawal 75 \
  --service-rate-payment 65 \
  --withdrawal-share 0.7 \
  --withdrawal-tellers 2 \
  --payment-tellers 1 \
  --horizon-hours 8 \
  --warmup-minutes 30 \
  --replications 500 \
  --plots \
  --output-dir outputs
```

Las gráficas se guardan en `outputs/` cuando se activa `--plots`.
