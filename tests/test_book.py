import unittest

from lob_sim.book import LimitOrderBook, Order, OrderType, Side


class LimitOrderBookTest(unittest.TestCase):
    def test_limit_order_crosses_and_trades_at_resting_price(self):
        book = LimitOrderBook()
        book.submit(Order("seller", Side.SELL, 10, OrderType.LIMIT, 101), step=1)

        trades = book.submit(Order("buyer", Side.BUY, 4, OrderType.LIMIT, 102), step=2)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 101)
        self.assertEqual(trades[0].quantity, 4)
        self.assertEqual(book.best_ask(), 101)

    def test_market_order_removes_multiple_levels(self):
        book = LimitOrderBook()
        book.submit(Order("seller_a", Side.SELL, 3, OrderType.LIMIT, 101), step=1)
        book.submit(Order("seller_b", Side.SELL, 5, OrderType.LIMIT, 102), step=1)

        trades = book.submit(Order("buyer", Side.BUY, 6, OrderType.MARKET), step=2)

        self.assertEqual([trade.quantity for trade in trades], [3, 3])
        self.assertEqual(book.best_ask(), 102)

    def test_cancel_removes_resting_order(self):
        book = LimitOrderBook()
        book.submit(Order("buyer", Side.BUY, 5, OrderType.LIMIT, 99), step=1)
        order_id = book.random_order_id()

        self.assertIsNotNone(order_id)
        self.assertTrue(book.cancel(order_id))
        self.assertIsNone(book.best_bid())


if __name__ == "__main__":
    unittest.main()
