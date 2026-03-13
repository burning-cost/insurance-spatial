"""
Tests for scores.py: non-conformity score functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_spatial.conformal.scores import (
    TweediePearsonScore,
    AbsoluteScore,
    PearsonScore,
    ScaledAbsoluteScore,
    make_score,
)


class TestTweediePearsonScore:
    def test_construction(self):
        s = TweediePearsonScore(power=1.5)
        assert s.power == 1.5
        assert s.name == "pearson_tweedie"

    def test_invalid_power_type_raises(self):
        with pytest.raises(TypeError):
            TweediePearsonScore(power="foo")

    def test_score_all_zeros_y(self):
        s = TweediePearsonScore(power=1.5)
        y = np.zeros(10)
        yhat = np.ones(10) * 200.0
        scores = s.score(y, yhat)
        assert np.all(scores > 0)
        # |0 - 200| / 200^0.75 = 200 / 200^0.75
        expected = 200.0 / (200.0 ** 0.75)
        np.testing.assert_allclose(scores, expected, rtol=1e-6)

    def test_score_perfect_prediction(self):
        s = TweediePearsonScore(power=1.5)
        y = np.array([100.0, 200.0, 300.0])
        yhat = y.copy()
        scores = s.score(y, yhat)
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)

    def test_score_nonnegative(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(2.0, 200.0, 100)
        yhat = rng.gamma(2.0, 200.0, 100)
        s = TweediePearsonScore(power=1.5)
        scores = s.score(y, yhat)
        assert np.all(scores >= 0)

    def test_score_shape_mismatch_raises(self):
        s = TweediePearsonScore(power=1.5)
        with pytest.raises(ValueError, match="same shape"):
            s.score(np.ones(5), np.ones(6))

    def test_score_zero_prediction_warns(self):
        s = TweediePearsonScore(power=1.5)
        with pytest.warns(UserWarning, match="<= 0"):
            s.score(np.ones(5), np.zeros(5))

    def test_invert_recovers_symmetric_interval(self):
        s = TweediePearsonScore(power=1.5)
        yhat = np.array([200.0, 500.0, 100.0])
        q = 1.5
        lo, hi = s.invert(yhat, q)
        margin = q * yhat ** 0.75
        np.testing.assert_allclose(hi, yhat + margin, rtol=1e-6)
        np.testing.assert_allclose(lo, np.maximum(yhat - margin, 0.0), rtol=1e-6)

    def test_invert_lower_non_negative(self):
        s = TweediePearsonScore(power=1.5)
        yhat = np.array([5.0])  # small prediction, large margin
        lo, hi = s.invert(yhat, threshold=1000.0)
        assert float(lo[0]) >= 0.0

    def test_score_invert_roundtrip(self):
        """Score then invert should recover the original y."""
        s = TweediePearsonScore(power=1.5)
        yhat = np.array([100.0, 300.0, 500.0])
        residuals = np.array([50.0, -100.0, 200.0])
        y = yhat + residuals
        scores = s.score(y, yhat)
        # Invert at the exact score value
        for i in range(len(y)):
            lo, hi = s.invert(yhat[i : i + 1], scores[i])
            assert float(lo[0]) <= float(y[i]) <= float(hi[0]) + 1e-6

    def test_different_powers(self):
        """Higher power should produce larger denom for same yhat, smaller score."""
        y = np.array([250.0])
        yhat = np.array([200.0])
        s1 = TweediePearsonScore(power=1.0)
        s2 = TweediePearsonScore(power=2.0)
        assert float(s1.score(y, yhat)[0]) > float(s2.score(y, yhat)[0])

    def test_poisson_special_case(self):
        """power=1 should give |y-yhat|/sqrt(yhat), same as PearsonScore."""
        y = np.array([150.0, 250.0])
        yhat = np.array([200.0, 200.0])
        s_tweedie = TweediePearsonScore(power=1.0)
        s_pearson = PearsonScore()
        np.testing.assert_allclose(
            s_tweedie.score(y, yhat), s_pearson.score(y, yhat), rtol=1e-6
        )


class TestAbsoluteScore:
    def test_name(self):
        assert AbsoluteScore().name == "absolute"

    def test_basic_score(self):
        s = AbsoluteScore()
        y = np.array([100.0, 200.0])
        yhat = np.array([80.0, 250.0])
        scores = s.score(y, yhat)
        np.testing.assert_allclose(scores, [20.0, 50.0], rtol=1e-6)

    def test_score_nonnegative(self):
        rng = np.random.default_rng(1)
        y = rng.exponential(100.0, 50)
        yhat = rng.exponential(100.0, 50)
        assert np.all(AbsoluteScore().score(y, yhat) >= 0)

    def test_invert_symmetric(self):
        s = AbsoluteScore()
        yhat = np.array([200.0, 500.0])
        lo, hi = s.invert(yhat, threshold=50.0)
        np.testing.assert_allclose(hi, [250.0, 550.0], rtol=1e-6)
        np.testing.assert_allclose(lo, [150.0, 450.0], rtol=1e-6)

    def test_invert_lower_clamped_at_zero(self):
        s = AbsoluteScore()
        lo, hi = s.invert(np.array([10.0]), threshold=200.0)
        assert float(lo[0]) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            AbsoluteScore().score(np.ones(3), np.ones(4))


class TestPearsonScore:
    def test_name(self):
        assert PearsonScore().name == "pearson"

    def test_score_formula(self):
        s = PearsonScore()
        y = np.array([150.0])
        yhat = np.array([100.0])
        expected = 50.0 / np.sqrt(100.0)
        np.testing.assert_allclose(s.score(y, yhat), expected, rtol=1e-6)

    def test_invert_uses_sqrt_yhat(self):
        s = PearsonScore()
        yhat = np.array([400.0])
        q = 2.0
        lo, hi = s.invert(yhat, q)
        expected_margin = 2.0 * np.sqrt(400.0)
        assert float(hi[0]) == pytest.approx(float(yhat[0]) + expected_margin, rel=1e-6)


class TestScaledAbsoluteScore:
    def test_construction_without_model_raises(self):
        with pytest.raises(TypeError, match="predict"):
            ScaledAbsoluteScore("not_a_model")

    def test_score_with_sigma(self, spread_model):
        s = ScaledAbsoluteScore(spread_model)
        y = np.array([150.0, 200.0])
        yhat = np.array([100.0, 200.0])
        sigma = np.array([50.0, 100.0])
        scores = s.score(y, yhat, sigma=sigma)
        expected = np.abs(y - yhat) / sigma
        np.testing.assert_allclose(scores, expected, rtol=1e-6)

    def test_score_without_x_or_sigma_raises(self, spread_model):
        s = ScaledAbsoluteScore(spread_model)
        with pytest.raises(ValueError, match="X or sigma"):
            s.score(np.ones(5), np.ones(5))

    def test_name(self, spread_model):
        assert ScaledAbsoluteScore(spread_model).name == "scaled_absolute"

    def test_invert_with_sigma(self, spread_model):
        s = ScaledAbsoluteScore(spread_model)
        yhat = np.array([200.0])
        sigma = np.array([50.0])
        q = 2.0
        lo, hi = s.invert(yhat, q, sigma=sigma)
        assert float(hi[0]) == pytest.approx(200.0 + 2.0 * 50.0)
        assert float(lo[0]) == pytest.approx(200.0 - 2.0 * 50.0)


class TestMakeScore:
    def test_pearson_tweedie(self):
        fn = make_score("pearson_tweedie", tweedie_power=1.5)
        assert isinstance(fn, TweediePearsonScore)
        assert fn.power == 1.5

    def test_pearson(self):
        fn = make_score("pearson")
        assert isinstance(fn, PearsonScore)

    def test_absolute(self):
        fn = make_score("absolute")
        assert isinstance(fn, AbsoluteScore)

    def test_scaled_absolute_requires_spread_model(self):
        with pytest.raises(ValueError, match="spread_model is required"):
            make_score("scaled_absolute")

    def test_scaled_absolute_with_model(self, spread_model):
        fn = make_score("scaled_absolute", spread_model=spread_model)
        assert isinstance(fn, ScaledAbsoluteScore)

    def test_unknown_score_raises(self):
        with pytest.raises(ValueError, match="Unknown nonconformity"):
            make_score("not_a_score")
