"""Numerical and isolation checks for the supplementary replacement maps."""
import numpy as np
from scipy.optimize._numdiff import approx_derivative

from epcrc.judge import JudgeResponses
from supplementary.exp02_reconstruction.run import (
    TargetMap, bootstrap_difference, evaluate, fit_reconstructors,
    fit_gate, gate_objective, predict_all, target_risk,
)


def toy(seed, n=12):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(2):
        peers = rng.dirichlet(np.ones(3), size=(n, 2))
        target = 0.3 * peers[:, 0] + 0.7 * peers[:, 1]
        blocks.append(np.concatenate((peers, target[:, None]), axis=1))
    return JudgeResponses(blocks)


def test_gate_gradient_matches_finite_difference():
    rng = np.random.default_rng(8)
    peers = rng.dirichlet(np.ones(3), size=(9, 2))
    truth = rng.dirichlet(np.ones(3), size=9)
    features = np.column_stack((np.ones(9), rng.normal(size=(9, 3))))
    theta = rng.normal(scale=0.2, size=8)
    args = (features, peers, truth, [slice(0, 4), slice(4, 9)], 0.03)
    _, analytic = gate_objective(theta, *args)
    numeric = approx_derivative(lambda x: gate_objective(x, *args)[0], theta).ravel()
    np.testing.assert_allclose(analytic, numeric, atol=2e-7, rtol=2e-5)


def test_predictions_cannot_read_removed_endpoint_and_are_probabilities():
    fit, cert, test = toy(1), toy(2), toy(3)
    maps, diagnostics, _ = fit_reconstructors(fit, cert, [0, 1], regularizations=(0.01,), maxiter=30)
    changed_blocks = [block.copy() for block in test.blocks]
    for block in changed_blocks:
        block[:, 2] = [1, 0, 0]
    changed = JudgeResponses(changed_blocks)
    for method, models in maps.items():
        first, second = predict_all(test, [0, 1], models), predict_all(changed, [0, 1], models)
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b)
            np.testing.assert_allclose(a.sum(axis=-1), 1., atol=1e-8)
            assert a.min() >= 0
        metrics, _ = evaluate(test, first, [0, 1])
        assert metrics["off_simplex_fraction"] == 0
    assert diagnostics["nonlinear_gate"][2]["trials"][0]["message"]


def test_exact_convex_target_is_recovered():
    fit, cert, test = toy(1), toy(2), toy(3)
    maps, _, _ = fit_reconstructors(fit, cert, [0, 1], regularizations=(0.01,), maxiter=20)
    metrics, _ = evaluate(test, predict_all(test, [0, 1], maps["global_convex"]), [0, 1])
    assert metrics["worst_endpoint_context_tv"] < 1e-7


def test_cluster_bootstrap_same_predictions_has_zero_interval():
    loss = [np.arange(12).reshape(6, 2) / 12., np.ones((6, 2)) * 0.1]
    result = bootstrap_difference(loss, loss, np.array([0, 0, 1, 1, 2, 2]), 40, 4)
    assert result["percentile_95_ci"] == [0., 0.]
    assert result["n_base_items"] == 3


def test_temperature_keeps_simplex_and_retained_self_reconstruction():
    responses = toy(3)
    model = TargetMap(weights=np.array([0.3, 0.7]), temperature=0.5)
    prediction = predict_all(responses, [0, 1], {2: model})
    for truth, pred in zip(responses.blocks, prediction):
        np.testing.assert_array_equal(truth[:, :2], pred[:, :2])
        np.testing.assert_allclose(pred.sum(-1), 1.)


def test_gate_learns_a_retained_output_dependent_choice():
    x = np.linspace(0.05, 0.95, 80)
    peer0 = np.column_stack((x, 1 - x, np.zeros(len(x))))
    peer1 = np.tile([0, 0, 1.], (len(x), 1))
    target = np.where((x > 0.5)[:, None], peer0, peer1)
    responses = JudgeResponses([np.stack((peer0, peer1, target), axis=1)])
    baseline = TargetMap(weights=np.array([0.5, 0.5]))
    model, diagnostic = fit_gate(responses, [0, 1], 2, baseline.weights, 0.001, 150)
    assert target_risk(responses, [0, 1], 2, model) < 0.15
    assert target_risk(responses, [0, 1], 2, baseline) > 0.45
    assert diagnostic["converged"]
