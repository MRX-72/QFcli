"""Tests for the comparison logic (pure dict math, no network)."""

from quant_finance.comparison import (
    determine_winners,
    calculate_overall_score,
    generate_recommendation
)


def _stock(ticker, cum=0.10, vol=0.20, sharpe=1.0, mdd=-0.15, roc=0.05,
           bullish=2, total=3, risk="MODERATE"):
    return {
        'ticker': ticker,
        'company_name': f"{ticker} Inc",
        'current_price': 100.0,
        'cumulative_return': cum,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'roc': roc,
        'bullish_count': bullish,
        'total_signals': total,
        'risk_score': risk,
        'sma_values': {20: 95.0, 50: 90.0, 200: 85.0},
        'signals': {'20': 'BULLISH', '50': 'BULLISH'},
    }


class TestDetermineWinners:
    def test_each_metric_picks_higher_value(self):
        a = _stock("A", cum=0.20, vol=0.15, sharpe=2.0, mdd=-0.05, roc=0.10, bullish=3, risk="STABLE")
        b = _stock("B", cum=0.10, vol=0.30, sharpe=0.5, mdd=-0.30, roc=-0.05, bullish=1, risk="HIGH RISK")
        winners = determine_winners(a, b)

        assert winners['cumulative_return'] == "A"
        assert winners['volatility'] == "A"          # lower is better for vol
        assert winners['sharpe_ratio'] == "A"
        assert winners['max_drawdown'] == "A"        # closer to 0 is better
        assert winners['roc'] == "A"
        assert winners['bullish_signals'] == "A"
        assert winners['risk_score'] == "A"          # STABLE > MODERATE > HIGH RISK

    def test_volatility_picks_lower(self):
        a = _stock("A", vol=0.15)
        b = _stock("B", vol=0.25)
        assert determine_winners(a, b)['volatility'] == "A"

    def test_max_drawdown_closer_to_zero(self):
        a = _stock("A", mdd=-0.05)
        b = _stock("B", mdd=-0.40)
        assert determine_winners(a, b)['max_drawdown'] == "A"

    def test_ties_reported(self):
        a = _stock("A", cum=0.10, vol=0.20, sharpe=1.0, mdd=-0.15,
                   roc=0.05, bullish=2, risk="MODERATE")
        b = _stock("B", cum=0.10, vol=0.20, sharpe=1.0, mdd=-0.15,
                   roc=0.05, bullish=2, risk="MODERATE")
        winners = determine_winners(a, b)
        assert winners['cumulative_return'] == "TIE"
        assert winners['volatility'] == "TIE"


class TestCalculateOverallScore:
    def test_scores_above_average(self):
        a = _stock("A", cum=0.20, vol=0.10, sharpe=2.0, mdd=-0.02, roc=0.2, bullish=3, risk="STABLE")
        b = _stock("B", cum=0.05, vol=0.40, sharpe=0.2, mdd=-0.50, roc=-0.2, bullish=0, risk="HIGH RISK")
        winners = determine_winners(a, b)
        assert calculate_overall_score(a, winners) > calculate_overall_score(b, winners)


class TestGenerateRecommendation:
    def test_clear_winner(self):
        a = _stock("A", cum=0.20, vol=0.10, sharpe=2.0, mdd=-0.02, roc=0.2, bullish=3, risk="STABLE")
        b = _stock("B", cum=0.05, vol=0.40, sharpe=0.2, mdd=-0.50, roc=-0.2, bullish=0, risk="HIGH RISK")
        winners = determine_winners(a, b)
        rec = generate_recommendation(a, b, winners)
        assert rec['overall_winner'] == "A"

    def test_strengths_are_not_empty_for_winner(self):
        a = _stock("A", cum=0.20, vol=0.10, sharpe=2.0, mdd=-0.02, roc=0.2, bullish=3, risk="STABLE")
        b = _stock("B", cum=0.05, vol=0.40, sharpe=0.2, mdd=-0.50, roc=-0.2, bullish=0, risk="HIGH RISK")
        winners = determine_winners(a, b)
        rec = generate_recommendation(a, b, winners)
        assert rec['strengths1']

    def test_score_reflects_win_count(self):
        a = _stock("A", cum=0.20, vol=0.10, sharpe=2.0, mdd=-0.02, roc=0.2, bullish=3, risk="STABLE")
        b = _stock("B", cum=0.05, vol=0.40, sharpe=0.2, mdd=-0.50, roc=-0.2, bullish=0, risk="HIGH RISK")
        winners = determine_winners(a, b)
        score1 = calculate_overall_score(a, winners)
        score2 = calculate_overall_score(b, winners)
        assert score1 + score2 == len(winners)