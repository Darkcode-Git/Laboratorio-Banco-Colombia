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


if __name__ == "__main__":
    unittest.main()
