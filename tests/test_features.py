import unittest

from lob_sim.book import BookSnapshot, Side, Trade
from lob_sim.features import compute_features


class MicrostructureFeaturesTest(unittest.TestCase):
    def test_microprice_moves_toward_thinner_side(self):
        snapshot = BookSnapshot(
            step=1,
            best_bid=99,
            best_ask=101,
            mid_price=100.0,
            spread=2,
            bid_depth=30,
            ask_depth=10,
            imbalance=0.5,
            last_trade_price=None,
        )

        features = compute_features(snapshot, [])

        self.assertEqual(features.microprice, 100.5)
        self.assertEqual(features.microprice_deviation, 0.5)

    def test_order_flow_imbalance_uses_aggressor_side(self):
        snapshot = BookSnapshot(1, 99, 101, 100.0, 2, 10, 10, 0.0, None)
        trades = [
            Trade(1, 101, 7, "b", "s", Side.BUY),
            Trade(1, 99, 3, "b", "s", Side.SELL),
        ]

        features = compute_features(snapshot, trades)

        self.assertEqual(features.signed_trade_volume, 4)
        self.assertEqual(features.order_flow_imbalance, 0.4)


if __name__ == "__main__":
    unittest.main()

