"""Simulación DES M/M/1 para el problema del Banco de Colombia."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TellerConfig:
    name: str
    arrival_rate: float  # clientes por minuto
    service_rate: float  # clientes por minuto


def validate_stability(arrival_rate: float, service_rate: float, name: str) -> float:
    if arrival_rate < 0:
        raise ValueError(f"La tasa de llegada de {name} no puede ser negativa.")
    if service_rate <= 0:
        raise ValueError(f"La tasa de servicio de {name} debe ser mayor a 0.")
    rho = arrival_rate / service_rate if arrival_rate > 0 else 0.0
    if rho >= 1:
        raise ValueError(
            f"Sistema inestable para {name}: rho={rho:.3f} (lambda/mu >= 1)."
        )
    return rho


def generate_arrival_times(
    arrival_rate: float, horizon: float, rng: np.random.Generator
) -> np.ndarray:
    if arrival_rate <= 0 or horizon <= 0:
        return np.array([], dtype=float)
    expected = max(int(arrival_rate * horizon), 1)
    batch_size = max(100, int(expected * 1.3) + 10)
    arrivals = []
    total = 0.0
    while total < horizon:
        interarrivals = rng.exponential(1 / arrival_rate, size=batch_size)
        times = total + np.cumsum(interarrivals)
        arrivals.append(times)
        total = times[-1]
    arrival_times = np.concatenate(arrivals)
    return arrival_times[arrival_times <= horizon]


def compute_waiting_times(
    interarrivals: np.ndarray, service_times: np.ndarray
) -> np.ndarray:
    n = service_times.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    increments = service_times[:-1] - interarrivals[1:]
    cumulative = np.cumsum(increments)
    cumulative_with_zero = np.concatenate(([0.0], cumulative))
    min_prefix = np.minimum.accumulate(cumulative_with_zero)
    waiting_times = cumulative_with_zero - min_prefix
    return waiting_times


def time_average_ncs(
    arrival_times: np.ndarray,
    departure_times: np.ndarray,
    start_time: float,
    end_time: float,
) -> float:
    if arrival_times.size == 0 or end_time <= start_time:
        return 0.0
    events = np.concatenate([arrival_times, departure_times])
    deltas = np.concatenate([np.ones(arrival_times.size), -np.ones(departure_times.size)])
    order = np.lexsort((deltas < 0, events))
    events = events[order]
    deltas = deltas[order]
    ncs = np.cumsum(deltas)
    if np.min(ncs) < 0:
        raise ValueError("NCS negativo detectado; revise la lógica de eventos.")
    segment_starts = np.concatenate(([0.0], events))
    segment_ends = np.concatenate((events, [end_time]))
    levels = np.concatenate(([0.0], ncs))
    overlap_start = np.maximum(segment_starts, start_time)
    overlap_end = np.minimum(segment_ends, end_time)
    durations = np.clip(overlap_end - overlap_start, 0.0, None)
    area = np.sum(levels * durations)
    return area / (end_time - start_time)


def compute_utilization(
    service_starts: np.ndarray,
    departure_times: np.ndarray,
    start_time: float,
    end_time: float,
) -> float:
    if service_starts.size == 0 or end_time <= start_time:
        return 0.0
    overlap = np.minimum(departure_times, end_time) - np.maximum(
        service_starts, start_time
    )
    busy_time = np.clip(overlap, 0.0, None).sum()
    return busy_time / (end_time - start_time)


def compute_metrics(
    arrival_times: np.ndarray,
    service_times: np.ndarray,
    waiting_times: np.ndarray,
    service_starts: np.ndarray,
    departure_times: np.ndarray,
    warmup_time: float,
) -> Dict[str, float]:
    if arrival_times.size == 0:
        return {"Wq": 0.0, "Lq": 0.0, "utilization": 0.0}
    end_time = departure_times[-1]
    if end_time <= warmup_time:
        return {"Wq": 0.0, "Lq": 0.0, "utilization": 0.0}
    mask = arrival_times >= warmup_time
    wq = float(waiting_times[mask].mean()) if np.any(mask) else 0.0
    utilization = compute_utilization(
        service_starts, departure_times, warmup_time, end_time
    )
    avg_ncs = time_average_ncs(
        arrival_times, departure_times, warmup_time, end_time
    )
    lq = max(avg_ncs - utilization, 0.0)
    return {"Wq": wq, "Lq": lq, "utilization": utilization}


def simulate_mm1(
    config: TellerConfig,
    horizon: float,
    warmup_time: float,
    rng: np.random.Generator,
    return_series: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    validate_stability(config.arrival_rate, config.service_rate, config.name)
    arrival_times = generate_arrival_times(config.arrival_rate, horizon, rng)
    service_times = (
        rng.exponential(1 / config.service_rate, size=arrival_times.size)
        if arrival_times.size > 0
        else np.array([], dtype=float)
    )
    interarrivals = np.diff(np.concatenate(([0.0], arrival_times)))
    waiting_times = compute_waiting_times(interarrivals, service_times)
    service_starts = arrival_times + waiting_times
    departure_times = service_starts + service_times
    metrics = compute_metrics(
        arrival_times,
        service_times,
        waiting_times,
        service_starts,
        departure_times,
        warmup_time,
    )
    if not return_series:
        return metrics, None
    series = {
        "arrival_times": arrival_times,
        "waiting_times": waiting_times,
    }
    return metrics, series


def confidence_interval(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    mean = float(np.mean(samples))
    if samples.size < 2:
        return mean, 0.0
    std = float(np.std(samples, ddof=1))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=samples.size - 1))
    half_width = t_crit * std / np.sqrt(samples.size)
    return mean, half_width


def plot_cumulative_average(
    values: np.ndarray,
    warmup_index: Optional[int],
    output_path: Path,
    title: str,
) -> None:
    if values.size == 0:
        return
    cumulative_mean = np.cumsum(values) / np.arange(1, values.size + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(cumulative_mean, label="Promedio acumulado")
    if warmup_index is not None:
        plt.axvline(warmup_index, color="red", linestyle="--", label="Warm-up")
    plt.xlabel("Cliente")
    plt.ylabel("Tiempo promedio en cola (min)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_replications(
    config: TellerConfig,
    horizon: float,
    replications: int,
    warmup_time: float,
    seed: int,
    plot_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    metrics = {"Wq": [], "Lq": [], "utilization": []}
    for rep in range(replications):
        rep_seed = int(rng.integers(0, 2**32 - 1))
        rep_rng = np.random.default_rng(rep_seed)
        collect_series = plot_dir is not None and rep == 0
        result, series = simulate_mm1(
            config, horizon, warmup_time, rep_rng, return_series=collect_series
        )
        for key in metrics:
            metrics[key].append(result[key])
        if collect_series and series is not None:
            warmup_index = (
                int(np.searchsorted(series["arrival_times"], warmup_time))
                if warmup_time > 0
                else None
            )
            output_path = plot_dir / f"promedio_acumulado_{config.name}.png"
            plot_cumulative_average(
                series["waiting_times"],
                warmup_index,
                output_path,
                f"{config.name} - Promedio acumulado de Wq",
            )
    return {key: np.array(values, dtype=float) for key, values in metrics.items()}


def format_ci(mean: float, half_width: float) -> str:
    return f"{mean:.4f} ± {half_width:.4f}"


def build_teller_configs(
    total_arrival_rate: float,
    withdrawal_share: float,
    withdrawal_tellers: int,
    payment_tellers: int,
    service_rate_withdrawal: float,
    service_rate_payment: float,
) -> Tuple[TellerConfig, ...]:
    if withdrawal_tellers <= 0 or payment_tellers <= 0:
        raise ValueError("La cantidad de cajeros debe ser mayor a 0.")
    withdrawal_rate = total_arrival_rate * withdrawal_share / withdrawal_tellers
    payment_rate = total_arrival_rate * (1 - withdrawal_share) / payment_tellers
    configs = []
    for idx in range(withdrawal_tellers):
        configs.append(
            TellerConfig(
                name=f"Retiro-{idx + 1}",
                arrival_rate=withdrawal_rate,
                service_rate=service_rate_withdrawal,
            )
        )
    for idx in range(payment_tellers):
        configs.append(
            TellerConfig(
                name=f"Pago-{idx + 1}",
                arrival_rate=payment_rate,
                service_rate=service_rate_payment,
            )
        )
    return tuple(configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulación DES M/M/1 para cajeros del Banco de Colombia."
    )
    parser.add_argument("--arrival-rate", type=float, default=60.0)
    parser.add_argument("--service-rate-withdrawal", type=float, default=75.0)
    parser.add_argument("--service-rate-payment", type=float, default=65.0)
    parser.add_argument("--withdrawal-share", type=float, default=0.7)
    parser.add_argument("--withdrawal-tellers", type=int, default=2)
    parser.add_argument("--payment-tellers", type=int, default=1)
    parser.add_argument("--horizon-hours", type=float, default=8.0)
    parser.add_argument("--warmup-minutes", type=float, default=30.0)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizon_minutes = args.horizon_hours * 60.0
    warmup_minutes = args.warmup_minutes
    configs = build_teller_configs(
        total_arrival_rate=args.arrival_rate / 60.0,
        withdrawal_share=args.withdrawal_share,
        withdrawal_tellers=args.withdrawal_tellers,
        payment_tellers=args.payment_tellers,
        service_rate_withdrawal=args.service_rate_withdrawal / 60.0,
        service_rate_payment=args.service_rate_payment / 60.0,
    )
    plot_dir = None
    if args.plots:
        plot_dir = Path(args.output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        rho = validate_stability(config.arrival_rate, config.service_rate, config.name)
        results = run_replications(
            config,
            horizon_minutes,
            args.replications,
            warmup_minutes,
            args.seed,
            plot_dir=plot_dir,
        )
        print(f"\nCajero {config.name} (rho={rho:.3f})")
        for metric, values in results.items():
            mean, half_width = confidence_interval(values)
            print(f"  {metric}: {format_ci(mean, half_width)}")


if __name__ == "__main__":
    main()
