from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import random

Action = Literal["retiro", "pago"]
UserType = Literal["rapido", "normal", "lento", "muy_lento"]

USER_TYPES: Tuple[UserType, ...] = ("rapido", "normal", "lento", "muy_lento")
ACTIONS: Tuple[Action, ...] = ("retiro", "pago")

ACTION_PROB: Dict[Action, float] = {"retiro": 0.70, "pago": 0.30}
USER_TYPE_PROB: Dict[Action, Dict[UserType, float]] = {
    "retiro": {"rapido": 0.23, "normal": 0.40, "lento": 0.17, "muy_lento": 0.20},
    "pago": {"rapido": 0.10, "normal": 0.20, "lento": 0.30, "muy_lento": 0.40},
}
SERVICE_MEAN_MIN: Dict[Action, Dict[UserType, float]] = {
    "retiro": {"rapido": 1.0, "normal": 2.0, "lento": 3.0, "muy_lento": 4.0},
    "pago": {"rapido": 3.0, "normal": 3.0, "lento": 5.0, "muy_lento": 7.0},
}
ARRIVAL_MEAN_MIN: Dict[Action, Dict[UserType, float]] = {
    "retiro": {"rapido": 1.0, "normal": 2.0, "lento": 3.0, "muy_lento": 3.0},
    "pago": {"rapido": 1.0, "normal": 2.0, "lento": 3.0, "muy_lento": 4.0},
}


@dataclass(frozen=True)
class Scenario:
    name: str
    cashier_capabilities: Tuple[Optional[Action], ...]


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(name="3_cajas_mixtas", cashier_capabilities=(None, None, None)),
    Scenario(name="1_retiro_2_pagos", cashier_capabilities=("retiro", "pago", "pago")),
    Scenario(name="2_retiros_1_pago", cashier_capabilities=("retiro", "retiro", "pago")),
    Scenario(name="4_cajas_3_retiros_1_pago", cashier_capabilities=("retiro", "retiro", "retiro", "pago")),
)


def _validate_probabilities() -> None:
    if abs(sum(ACTION_PROB.values()) - 1.0) > 1e-9:
        raise ValueError("La probabilidad de acciones debe sumar 1.0")
    for action in ACTIONS:
        if abs(sum(USER_TYPE_PROB[action].values()) - 1.0) > 1e-9:
            raise ValueError(f"Las probabilidades de tipos para {action} deben sumar 1.0")


def _weighted_mean(values: Dict[UserType, float], probs: Dict[UserType, float]) -> float:
    return sum(values[user_type] * probs[user_type] for user_type in USER_TYPES)


def _joint_prob(action: Action, user_type: UserType) -> float:
    return ACTION_PROB[action] * USER_TYPE_PROB[action][user_type]


def _theoretical_total_arrival_rate() -> float:
    expected_interarrival = sum(
        _joint_prob(action, user_type) * ARRIVAL_MEAN_MIN[action][user_type]
        for action in ACTIONS
        for user_type in USER_TYPES
    )
    return 1.0 / expected_interarrival


def _action_service_rate(action: Action) -> float:
    expected_service = _weighted_mean(SERVICE_MEAN_MIN[action], USER_TYPE_PROB[action])
    return 1.0 / expected_service


def validate_scenario_stability(scenario: Scenario) -> Dict[str, float]:
    total_lambda = _theoretical_total_arrival_rate()
    lambda_by_action = {action: total_lambda * ACTION_PROB[action] for action in ACTIONS}
    mu_by_action = {action: _action_service_rate(action) for action in ACTIONS}

    rho_by_cashier: Dict[str, float] = {}
    mixed_cashiers = sum(1 for capability in scenario.cashier_capabilities if capability is None)
    specialized_counts = {
        action: sum(1 for capability in scenario.cashier_capabilities if capability == action) for action in ACTIONS
    }

    if mixed_cashiers:
        arrival_per_mixed = total_lambda / mixed_cashiers
        overall_service_mean = sum(
            _joint_prob(action, user_type) * SERVICE_MEAN_MIN[action][user_type]
            for action in ACTIONS
            for user_type in USER_TYPES
        )
        mu_mixed = 1.0 / overall_service_mean
        for i in range(mixed_cashiers):
            rho = arrival_per_mixed / mu_mixed
            rho_by_cashier[f"cajero_{i + 1}"] = rho
            if rho >= 1.0:
                raise ValueError(f"Escenario inestable {scenario.name}: rho={rho:.4f}")

    start_idx = mixed_cashiers + 1
    for action in ACTIONS:
        count = specialized_counts[action]
        if count == 0:
            continue
        arrival_per_cashier = lambda_by_action[action] / count
        rho = arrival_per_cashier / mu_by_action[action]
        for i in range(count):
            rho_by_cashier[f"cajero_{start_idx}"] = rho
            start_idx += 1
            if rho >= 1.0:
                raise ValueError(f"Escenario inestable {scenario.name}: rho={rho:.4f}")
    return rho_by_cashier


def _sample_action(rng: random.Random) -> Action:
    u = rng.random()
    return "retiro" if u < ACTION_PROB["retiro"] else "pago"


def _sample_user_type(rng: random.Random, action: Action) -> UserType:
    u = rng.random()
    cumulative = 0.0
    for user_type in USER_TYPES:
        cumulative += USER_TYPE_PROB[action][user_type]
        if u <= cumulative:
            return user_type
    return USER_TYPES[-1]


@dataclass
class CashierStats:
    customers: int = 0
    total_wait: float = 0.0
    total_service: float = 0.0
    total_system: float = 0.0


@dataclass
class ReplicaResult:
    scenario_name: str
    cashier_metrics: Dict[str, Dict[str, float]]
    type_counts: Dict[UserType, int]
    action_counts: Dict[Action, int]
    total_customers: int
    average_wait: float


def _allowed_cashiers(capabilities: Tuple[Optional[Action], ...], action: Action) -> List[int]:
    return [i for i, capability in enumerate(capabilities) if capability in (None, action)]


def simulate_replica(
    scenario: Scenario,
    horizon_minutes: float,
    seed: int,
) -> ReplicaResult:
    rng = random.Random(seed)
    available_at = [0.0 for _ in scenario.cashier_capabilities]
    cashier_stats = [CashierStats() for _ in scenario.cashier_capabilities]

    t = 0.0
    type_counts: Dict[UserType, int] = {user_type: 0 for user_type in USER_TYPES}
    action_counts: Dict[Action, int] = {action: 0 for action in ACTIONS}

    while True:
        action = _sample_action(rng)
        user_type = _sample_user_type(rng, action)
        interarrival = rng.expovariate(1.0 / ARRIVAL_MEAN_MIN[action][user_type])
        t += interarrival
        if t > horizon_minutes:
            break

        service_time = rng.expovariate(1.0 / SERVICE_MEAN_MIN[action][user_type])

        candidates = _allowed_cashiers(scenario.cashier_capabilities, action)
        cashier_index = min(candidates, key=lambda idx: available_at[idx])

        wait = max(0.0, available_at[cashier_index] - t)
        start_service = t + wait
        departure_time = start_service + service_time
        available_at[cashier_index] = departure_time

        stats = cashier_stats[cashier_index]
        stats.customers += 1
        stats.total_wait += wait
        stats.total_service += service_time
        stats.total_system += wait + service_time

        type_counts[user_type] += 1
        action_counts[action] += 1

    cashier_metrics: Dict[str, Dict[str, float]] = {}
    total_customers = 0
    total_wait = 0.0

    for idx, stats in enumerate(cashier_stats, start=1):
        if stats.customers:
            avg_wait = stats.total_wait / stats.customers
            avg_service = stats.total_service / stats.customers
            avg_system = stats.total_system / stats.customers
            lambda_hat = stats.customers / horizon_minutes
            utilization = min(stats.total_service / horizon_minutes, 1.0)
            l = lambda_hat * avg_system
            lq = lambda_hat * avg_wait
        else:
            avg_wait = avg_service = avg_system = lambda_hat = utilization = l = lq = 0.0

        cashier_metrics[f"cajero_{idx}"] = {
            "clientes": stats.customers,
            "tiempo_promedio_espera": avg_wait,
            "tiempo_promedio_servicio": avg_service,
            "tiempo_promedio_sistema": avg_system,
            "lambda": lambda_hat,
            "utilizacion": utilization,
            "L": l,
            "Lq": lq,
            "Wq": avg_wait,
            "W": avg_system,
        }
        total_customers += stats.customers
        total_wait += stats.total_wait

    average_wait = (total_wait / total_customers) if total_customers else 0.0
    return ReplicaResult(
        scenario_name=scenario.name,
        cashier_metrics=cashier_metrics,
        type_counts=type_counts,
        action_counts=action_counts,
        total_customers=total_customers,
        average_wait=average_wait,
    )


def confidence_interval_95(values: Iterable[float]) -> Tuple[float, float]:
    values_list = list(values)
    if not values_list:
        return (0.0, 0.0)
    if len(values_list) == 1:
        return (values_list[0], values_list[0])
    m = mean(values_list)
    margin = 1.96 * stdev(values_list) / sqrt(len(values_list))
    return (m - margin, m + margin)


def _aggregate_cashier_service_means(replica_results: List[ReplicaResult]) -> Dict[str, float]:
    all_cashiers = replica_results[0].cashier_metrics.keys()
    return {
        cashier: mean(result.cashier_metrics[cashier]["tiempo_promedio_servicio"] for result in replica_results)
        for cashier in all_cashiers
    }


def _aggregate_type_average(replica_results: List[ReplicaResult]) -> Dict[UserType, float]:
    return {
        user_type: mean(result.type_counts[user_type] for result in replica_results) for user_type in USER_TYPES
    }


def _replica_min_by_type(replica_results: List[ReplicaResult]) -> Dict[UserType, Dict[str, int]]:
    min_data: Dict[UserType, Dict[str, int]] = {}
    for user_type in USER_TYPES:
        min_replica_index, min_count = min(
            enumerate(replica_results, start=1), key=lambda item: item[1].type_counts[user_type]
        )
        min_data[user_type] = {"replica": min_replica_index, "usuarios": min_count.type_counts[user_type]}
    return min_data


def run_experiment(replicas: int = 600, horizon_minutes: float = 480.0, base_seed: int = 42) -> Dict[str, object]:
    _validate_probabilities()

    scenario_outputs: Dict[str, Dict[str, object]] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        rho = validate_scenario_stability(scenario)
        results = [
            simulate_replica(
                scenario=scenario,
                horizon_minutes=horizon_minutes,
                seed=base_seed + scenario_index * 10_000 + replica_index,
            )
            for replica_index in range(replicas)
        ]

        avg_waits = [result.average_wait for result in results]
        avg_wait_ci = confidence_interval_95(avg_waits)

        service_by_cashier = _aggregate_cashier_service_means(results)
        best_cashier = min(service_by_cashier, key=service_by_cashier.get)
        worst_cashier = max(service_by_cashier, key=service_by_cashier.get)

        scenario_outputs[scenario.name] = {
            "rho_por_cajero": rho,
            "punto_1_cajeros": {
                "tiempo_promedio_servicio_por_cajero": service_by_cashier,
                "cajero_menor_tiempo": best_cashier,
                "cajero_mayor_tiempo": worst_cashier,
            },
            "punto_2_promedio_usuarios_por_tipo": _aggregate_type_average(results),
            "punto_3_total_usuarios_por_tipo_por_replica": [result.type_counts for result in results],
            "punto_3_replica_menor_por_tipo": _replica_min_by_type(results),
            "metricas_globales": {
                "espera_promedio": mean(avg_waits),
                "ic95_espera_promedio": avg_wait_ci,
                "total_clientes_promedio": mean(result.total_customers for result in results),
                "utilizacion_promedio": mean(
                    mean(c["utilizacion"] for c in result.cashier_metrics.values()) for result in results
                ),
            },
        }

    wait_3_cashiers = min(
        scenario_outputs[name]["metricas_globales"]["espera_promedio"]
        for name in ["3_cajas_mixtas", "1_retiro_2_pagos", "2_retiros_1_pago"]
    )
    wait_4_cashiers = scenario_outputs["4_cajas_3_retiros_1_pago"]["metricas_globales"]["espera_promedio"]
    needs_new_cashier = wait_4_cashiers < (wait_3_cashiers * 0.9)

    optimal_scenario = min(
        scenario_outputs,
        key=lambda name: scenario_outputs[name]["metricas_globales"]["espera_promedio"],
    )

    return {
        "configuracion_optima": optimal_scenario,
        "necesita_nuevo_cajero": needs_new_cashier,
        "escenarios": scenario_outputs,
        "parametros": {
            "replicas": replicas,
            "horizonte_minutos": horizon_minutes,
            "semilla_base": base_seed,
        },
    }


def _plot_wait_comparison(results: Dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    escenarios = list(results["escenarios"].keys())
    esperas = [results["escenarios"][name]["metricas_globales"]["espera_promedio"] for name in escenarios]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(escenarios, esperas)
    ax.set_title("Comparación de espera promedio por escenario")
    ax.set_ylabel("Minutos")
    ax.set_xticklabels(escenarios, rotation=20, ha="right")
    fig.tight_layout()
    plt.savefig("comparacion_esperas.png", dpi=150)


def main() -> None:
    results = run_experiment()
    print("=== Resumen del laboratorio bancario ===")
    print(f"Configuración óptima: {results['configuracion_optima']}")
    print(f"¿Se recomienda nuevo cajero?: {results['necesita_nuevo_cajero']}")
    for scenario_name, data in results["escenarios"].items():
        metricas = data["metricas_globales"]
        print(
            f"- {scenario_name}: espera promedio={metricas['espera_promedio']:.4f} min "
            f"IC95={metricas['ic95_espera_promedio']}"
        )
    _plot_wait_comparison(results)


if __name__ == "__main__":
    main()
