"""Offline tests for the statistical factor model + Fama-French overlay."""

import numpy as np
import pandas as pd
import pytest

from quant_finance import factor_model as fm


def _factor_returns(seed=0, T=400, n_assets=5, n_factors=2, noise_std=0.3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="B")
    loadings = rng.normal(size=(n_assets, n_factors))
    factors = rng.normal(size=(T, n_factors))
    noise = rng.normal(0, noise_std, size=(T, n_assets))
    ret = pd.DataFrame(factors @ loadings.T + noise,
                       index=idx, columns=list("ABCDE"))
    return ret, loadings, factors


class TestPcaFactorModel:
    def test_recovers_k_factors(self):
        ret, _, _ = _factor_returns(seed=1)
        res = fm.pca_factor_model(ret, n_factors=2)
        assert res['n_factors'] == 2

    def test_explained_variance_reaches_high_level(self):
        ret, _, _ = _factor_returns(seed=2)
        res = fm.pca_factor_model(ret, min_expl_var=0.90)
        assert res['explained_variance'] >= 0.90
        assert 0.0 < res['explained_variance'] <= 1.0
        # more factors retained than a tiny cutoff
        small = fm.pca_factor_model(ret, min_expl_var=0.30)
        assert small['n_factors'] <= res['n_factors']

    def test_covariance_is_psd(self):
        ret, _, _ = _factor_returns(seed=3)
        res = fm.pca_factor_model(ret, n_factors=2)
        eigvals = np.linalg.eigvalsh(res['covariance'].to_numpy())
        assert eigvals.min() >= -1e-9

    def test_covariance_decomposition(self):
        ret, _, _ = _factor_returns(seed=4)
        res = fm.pca_factor_model(ret, n_factors=2)
        L = res['loadings'].to_numpy()
        rebuild = L @ np.diag(res['eigenvalues'][:2]) @ L.T
        idio = np.diag((res['idiosyncratic_std'] ** 2).to_numpy())
        np.testing.assert_allclose(res['common_covariance'].to_numpy(),
                                   rebuild, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(res['covariance'].to_numpy(),
                                   rebuild + idio, rtol=1e-6, atol=1e-12)

    def test_idiosyncratic_std_positive(self):
        ret, _, _ = _factor_returns(seed=5)
        res = fm.pca_factor_model(ret, n_factors=2)
        assert (res['idiosyncratic_std'] > 0).all()

    def test_factor_returns_unit_variance(self):
        ret, _, _ = _factor_returns(seed=6)
        res = fm.pca_factor_model(ret, n_factors=2)
        std = res['factor_returns'].std()
        assert std.abs().max() < 1.1

    def test_n_factors_capped_at_assets(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2022-01-01", periods=300, freq="B")
        ret = pd.DataFrame(rng.normal(0, 0.01, (300, 3)),
                           index=idx, columns=list("ABC"))
        res = fm.pca_factor_model(ret, n_factors=99)
        assert res['n_factors'] <= 3

    def test_requires_two_assets(self):
        idx = pd.date_range("2022-01-01", periods=100, freq="B")
        single = pd.DataFrame({"A": np.random.default_rng(0).normal(size=100)}, index=idx)
        with pytest.raises(ValueError):
            fm.pca_factor_model(single)

    def test_factor_model_cov_matches(self):
        ret, _, _ = _factor_returns(seed=8)
        assert fm.factor_model_cov(ret, n_factors=2).equals(
            fm.pca_factor_model(ret, n_factors=2)['covariance']
        )


class TestFamaFrenchOverlay:
    def test_parse_ff_csv_basic(self):
        text = (
            ", Mkt-RF, SMB, HML, RF\n"
            "20230103,0.10,-0.05,0.03,0.005\n"
            "20230104,-0.20,0.10,0.00,0.005\n"
            "20230105,0.15,0.05,-0.10,0.005\n"
        )
        df = fm.parse_ff_csv(text)
        assert list(df.columns) == ['Mkt-RF', 'SMB', 'HML', 'RF']
        assert len(df) == 3
        # percent values become decimals
        assert df['Mkt-RF'].iloc[0] == pytest.approx(0.001)
        assert df['RF'].iloc[0] == pytest.approx(0.00005)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0] == pd.Timestamp('2023-01-03')

    def test_parse_ff_csv_ignores_trailing_table(self):
        text = (
            ", Mkt-RF, SMB, HML, RF\n"
            "20230103,0.10,0.00,0.00,0.005\n"
            "20230104,0.20,0.00,0.00,0.005\n"
            "\n"
            "This file was created by CMPT_ME_BEME_RETS using the 202101 Daily CSV factor returns.\n"
        )
        df = fm.parse_ff_csv(text)
        assert len(df) == 2

    def test_parse_ff_csv_missing_header_raises(self):
        with pytest.raises(ValueError):
            fm.parse_ff_csv("garbage\nnot a factor file\n")

    def test_ff_betas_recover_true_betas(self):
        rng = np.random.default_rng(11)
        T = 500
        idx = pd.date_range("2022-01-01", periods=T, freq="B")
        fac = pd.DataFrame({
            'Mkt-RF': rng.normal(0.0005, 0.01, T),
            'SMB': rng.normal(0, 0.008, T),
            'HML': rng.normal(0, 0.008, T),
        }, index=idx)
        true_b = np.array([1.2, 0.5, -0.3])
        noise = rng.normal(0, 0.005, T)
        asset = true_b @ fac.T.values + noise + 0.0001
        ret = pd.DataFrame({"X": asset}, index=idx)
        fit = fm.ff_betas(ret, fac)
        got = fit['betas']['X'].to_numpy()
        np.testing.assert_allclose(got, true_b, atol=0.15)
        # intercept should recover the daily alpha we injected
        assert fit['alpha_annual']['X'] == pytest.approx(0.0001 * 252, abs=0.05)

    def test_ff_betas_insufficient_overlap_raises(self):
        ret = pd.DataFrame({"X": np.ones(50)}, index=pd.date_range("2022-01-01", periods=50, freq="B"))
        fac = pd.DataFrame({"Mkt-RF": np.ones(100)},
                           index=pd.date_range("2021-01-01", periods=100, freq="B"))
        with pytest.raises(ValueError):
            fm.ff_betas(ret, fac)

    def test_factor_premia_annualizes(self):
        idx = pd.date_range("2022-01-01", periods=252, freq="B")
        fac = pd.DataFrame({"Mkt-RF": np.full(252, 0.001), "RF": np.zeros(252)}, index=idx)
        premia = fm.factor_premia(fac)
        assert 'RF' not in premia.index
        assert premia['Mkt-RF'] == pytest.approx(0.252)

    def test_ff_expected_returns_matches_betas_times_premia(self):
        rng = np.random.default_rng(13)
        T = 500
        idx = pd.date_range("2022-01-01", periods=T, freq="B")
        fac = pd.DataFrame({'Mkt-RF': rng.normal(0.0005, 0.01, T), 'SMB': rng.normal(0, 0.01, T)},
                           index=idx)
        ret = pd.DataFrame({
            'A': 1.0 * fac['Mkt-RF'] + 0.5 * fac['SMB'] + rng.normal(0, 0.005, T),
            'B': rng.normal(0, 0.01, T),
        }, index=idx)
        premia = pd.Series({'Mkt-RF': 0.08, 'SMB': 0.03})
        expected = fm.ff_expected_returns(ret, fac, premia=premia, rf=0.0)
        fit = fm.ff_betas(ret, fac)
        manual = (fit['betas'].T @ premia).rename('e')
        np.testing.assert_allclose(expected, manual, atol=1e-12)
        assert expected.index.equals(ret.columns)

    def test_ff_rf_column_not_a_factor_and_is_subtracted(self):
        # 'RF' must not appear among betas; it is subtracted from the LHS so a
        # synthetically RF-excess-return asset recovers its true gross returns.
        rng = np.random.default_rng(2)
        T = 400
        idx = pd.date_range("2023-01-02", periods=T, freq="B")
        rf = np.linspace(0.0001, 0.0002, T)
        fac = pd.DataFrame({
            'Mkt-RF': rng.normal(0.0005, 0.01, T),
            'RF': rf,
        }, index=idx)
        betas_true = {'Mkt-RF': 1.2}
        ret = pd.DataFrame({
            'A': rf + 1.2 * fac['Mkt-RF'].to_numpy() + rng.normal(0, 0.005, T),
        }, index=idx)
        out = fm.ff_betas(ret, fac)
        assert list(out['betas'].index) == ['Mkt-RF']
        assert out['betas'].loc['Mkt-RF', 'A'] == pytest.approx(betas_true['Mkt-RF'], abs=0.05)

    def test_ff_betas_accept_tz_aware_returns(self):
        # yfinance caches prices with a tz-aware index; factors are naive, so
        # the overlay used to find zero overlap and fail. The intersection must
        # align on calendar dates regardless of the index's timezone.
        rng = np.random.default_rng(7)
        T = 300
        idx = pd.date_range("2023-01-01", periods=T, freq="B", tz="America/New_York")
        fac_idx = pd.date_range("2023-01-01", periods=T, freq="B")
        fac = pd.DataFrame({'Mkt-RF': rng.normal(0.0005, 0.01, T)}, index=fac_idx)
        ret = pd.DataFrame({'A': 0.8 * fac['Mkt-RF'].to_numpy() + rng.normal(0, 0.005, T)},
                           index=idx)
        out = fm.ff_betas(ret, fac)
        assert np.isfinite(out['betas'].to_numpy()).all()
        assert abs(out['betas'].loc['Mkt-RF', 'A'] - 0.8) < 0.1