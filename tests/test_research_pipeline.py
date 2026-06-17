import unittest

import pandas as pd

from lob_sim.config import ResearchConfig
from lob_sim.data import generate_event_log
from lob_sim.labels import construct_passive_orders, expected_posting_value, label_orders, signed_markout
from lob_sim.models import chronological_split
from lob_sim.phd_layer import PhDProfile


class ResearchPipelineTest(unittest.TestCase):
    def test_side_adjusted_markout_sign(self):
        self.assertEqual(signed_markout("buy", 100, 103, 1), 3)
        self.assertEqual(signed_markout("buy", 100, 98, 1), -2)
        self.assertEqual(signed_markout("sell", 100, 97, 1), 3)
        self.assertEqual(signed_markout("sell", 100, 102, 1), -2)

    def test_expected_value_calculation(self):
        value = expected_posting_value(fill_probability=0.5, spread_ticks=2, expected_markout=-0.25, fees_ticks=0.05)
        self.assertAlmostEqual(value, 0.35)

    def test_fill_detection_and_censoring(self):
        config = ResearchConfig(fill_horizon=3, markout_horizons=[1], queue_ahead_fraction=0.0, queue_ahead_max=0, order_size=5)
        events = pd.DataFrame(
            [
                {"step": 1, "best_bid": 99, "best_ask": 101, "mid_price": 100, "markout_mid_price": 100, "spread": 2, "spread_bps": 2, "bid_depth": 10, "ask_depth": 10, "queue_imbalance": 0, "microprice_deviation": 0, "recent_signed_trade_flow": 0, "recent_trade_volume": 0, "recent_volatility": 0, "recent_mid_move": 0, "sell_trade_qty": 0, "buy_trade_qty": 0, "trade_price": float("nan"), "cancel_side": "", "cancel_price": float("nan"), "cancel_qty": 0},
                {"step": 2, "best_bid": 99, "best_ask": 101, "mid_price": 100, "markout_mid_price": 100, "spread": 2, "spread_bps": 2, "bid_depth": 10, "ask_depth": 10, "queue_imbalance": 0, "microprice_deviation": 0, "recent_signed_trade_flow": 0, "recent_trade_volume": 5, "recent_volatility": 0, "recent_mid_move": 0, "sell_trade_qty": 5, "buy_trade_qty": 0, "trade_price": 99, "cancel_side": "", "cancel_price": float("nan"), "cancel_qty": 0},
                {"step": 3, "best_bid": 99, "best_ask": 101, "mid_price": 100, "markout_mid_price": 98, "spread": 2, "spread_bps": 2, "bid_depth": 10, "ask_depth": 10, "queue_imbalance": 0, "microprice_deviation": 0, "recent_signed_trade_flow": -5, "recent_trade_volume": 5, "recent_volatility": 0, "recent_mid_move": -2, "sell_trade_qty": 0, "buy_trade_qty": 0, "trade_price": float("nan"), "cancel_side": "", "cancel_price": float("nan"), "cancel_qty": 0},
            ]
        )
        orders = construct_passive_orders(events.iloc[[0]], config)
        buy_order = orders[orders["side"] == "buy"]
        labeled = label_orders(buy_order, events, config).iloc[0]
        self.assertEqual(labeled["filled"], 1)
        self.assertEqual(labeled["time_to_fill"], 1)
        self.assertEqual(labeled["markout_1"], -1)

    def test_chronological_split_has_no_overlapping_steps(self):
        df = pd.DataFrame({"step": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "side": ["buy", "sell"] * 5})
        train, validation, test = chronological_split(df, 0.6, 0.2)
        self.assertLess(train["step"].max(), validation["step"].min())
        self.assertLess(validation["step"].max(), test["step"].min())

    def test_deterministic_generation_under_fixed_seed(self):
        config = ResearchConfig(steps=100, seed=123)
        profile = PhDProfile.from_json("configs/phd_profile.json")
        first = generate_event_log(config, profile)
        second = generate_event_log(config, profile)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()

