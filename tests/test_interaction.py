import unittest

import pandas as pd

from lob_sim.interaction import (
    add_flow_depletion_features,
    aggressive_flow_against_passive,
    queue_depletion_components,
    signed_flow_persistence,
)
from lob_sim.null_model import local_shuffle_trade_signs
from lob_sim.real_data import load_btc_events, normalize_btc_events


class InteractionFeatureTest(unittest.TestCase):
    def test_passive_side_aggressive_flow_sign(self):
        self.assertEqual(aggressive_flow_against_passive("buy", buy_trade_qty=0, sell_trade_qty=5), 5)
        self.assertEqual(aggressive_flow_against_passive("sell", buy_trade_qty=5, sell_trade_qty=0), 5)
        self.assertEqual(aggressive_flow_against_passive("buy", buy_trade_qty=5, sell_trade_qty=0), -5)

    def test_signed_flow_persistence(self):
        buy = pd.Series([0, 0, 3])
        sell = pd.Series([1, 2, 0])
        self.assertEqual(signed_flow_persistence("buy", buy, sell), 0.0)
        self.assertEqual(signed_flow_persistence("sell", buy, sell), 0.0)
        self.assertGreater(signed_flow_persistence("buy", pd.Series([0, 0]), pd.Series([2, 3])), 0)

    def test_queue_depletion_components(self):
        events = pd.DataFrame(
            {
                "bid_depth": [100, 80],
                "ask_depth": [100, 100],
                "sell_trade_qty": [10, 5],
                "buy_trade_qty": [0, 0],
                "cancel_side": ["buy", ""],
                "cancel_qty": [5, 0],
            }
        )
        components = queue_depletion_components("buy", events, current_depth=80)
        self.assertAlmostEqual(components["total_depletion"], 0.2)
        self.assertAlmostEqual(components["trade_depletion"], 0.15)
        self.assertAlmostEqual(components["cancel_depletion"], 0.05)

    def test_interaction_feature_added(self):
        events = pd.DataFrame(
            {
                "step": [1, 2, 3],
                "buy_trade_qty": [0, 0, 0],
                "sell_trade_qty": [2, 3, 0],
                "trade_qty": [2, 3, 0],
                "bid_depth": [100, 95, 95],
                "ask_depth": [100, 100, 100],
                "cancel_side": ["", "", ""],
                "cancel_qty": [0, 0, 0],
            }
        )
        orders = pd.DataFrame({"step": [3], "side": ["buy"], "bid_depth": [95], "ask_depth": [100]})
        featured = add_flow_depletion_features(orders, events, [2])
        self.assertIn("flow_depletion_interaction_2", featured)
        self.assertGreater(featured.loc[0, "flow_persistence_2"], 0)

    def test_local_shuffle_is_deterministic(self):
        events = pd.DataFrame({"buy_trade_qty": [1, 0, 2, 0], "sell_trade_qty": [0, 1, 0, 2], "trade_qty": [1, 1, 2, 2]})
        first = local_shuffle_trade_signs(events, block_size=2, seed=7)
        second = local_shuffle_trade_signs(events, block_size=2, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_btc_mock_schema_loads(self):
        raw = load_btc_events("data/fixtures/btc_usd_mock_events.csv")
        normalized = normalize_btc_events(raw)
        self.assertIn("mid_price", normalized)
        self.assertTrue((normalized["spread"] > 0).all())


if __name__ == "__main__":
    unittest.main()

