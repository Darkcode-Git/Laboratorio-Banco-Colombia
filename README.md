# Laboratorio-Banco-Colombia

Laboratorio: Problema Bancario. Simulación de sistemas de colas M/M/1 para análisis y optimización del Banco de Colombia.

## Requisitos

- Node.js 18+

## Instalación

No hay dependencias externas. Solo se necesita Node.js.

## Ejecución

```bash
npm start -- --tellers 4 --hours 8 --arrival-rate 20 --service-rate 24 --payment-prob 0.55
```

También puedes ejecutar directamente:

```bash
node src/index.js --help
```

## Parámetros principales

- `--tellers`: cantidad de cajeros.
- `--hours`: horas de simulación.
- `--arrival-rate`: tasa de llegadas por hora.
- `--service-rate`: tasa de servicio por hora.
- `--payment-prob`: probabilidad de usuarios de pago.
- `--replications`: réplicas iniciales.
- `--relative-error`: error relativo objetivo para calcular réplicas necesarias.
- `--wait-threshold`: umbral de espera promedio (minutos) para recomendar un nuevo cajero.

## Evidencias de resultados

Un ejemplo de salida se encuentra en `results/sample_output.txt`.
