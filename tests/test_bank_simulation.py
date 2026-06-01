import unittest

from bank_simulation import ACTION_PROB, SCENARIOS, USER_TYPE_PROB, run_experiment, validate_scenario_stability


class BankSimulationTests(unittest.TestCase):
    def test_probabilities_are_valid(self) -> None:
        self.assertAlmostEqual(sum(ACTION_PROB.values()), 1.0, places=9)
        for action, probs in USER_TYPE_PROB.items():
            with self.subTest(action=action):
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_all_scenarios_are_stable(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.name):
                rho_by_cashier = validate_scenario_stability(scenario)
                self.assertTrue(all(rho < 1.0 for rho in rho_by_cashier.values()))

    def test_experiment_returns_required_outputs(self) -> None:
        result = run_experiment(replicas=10, horizon_minutes=120, base_seed=123)
        self.assertIn("configuracion_optima", result)
        self.assertIn("necesita_nuevo_cajero", result)
        self.assertIn("escenarios", result)

        escenarios = result["escenarios"]
        self.assertEqual(set(escenarios.keys()), {scenario.name for scenario in SCENARIOS})
        for scenario_name, data in escenarios.items():
            with self.subTest(scenario=scenario_name):
                self.assertIn("punto_1_cajeros", data)
                self.assertIn("punto_2_promedio_usuarios_por_tipo", data)
                self.assertIn("punto_3_total_usuarios_por_tipo_por_replica", data)
                self.assertIn("metricas_globales", data)
                self.assertEqual(len(data["punto_3_total_usuarios_por_tipo_por_replica"]), 10)

    def test_experiment_handles_larger_replica_count(self) -> None:
        result = run_experiment(replicas=30, horizon_minutes=120, base_seed=456)
        for data in result["escenarios"].values():
            self.assertEqual(len(data["punto_3_total_usuarios_por_tipo_por_replica"]), 30)
            ci_low, ci_high = data["metricas_globales"]["ic95_espera_promedio"]
            self.assertLessEqual(ci_low, ci_high)
            self.assertGreaterEqual(data["metricas_globales"]["espera_promedio"], 0.0)

    def test_experiment_production_like_parameters(self) -> None:
        result = run_experiment(replicas=60, horizon_minutes=480, base_seed=999)
        self.assertIn(result["configuracion_optima"], result["escenarios"])
        self.assertIsInstance(result["necesita_nuevo_cajero"], bool)


if __name__ == "__main__":
    unittest.main()
