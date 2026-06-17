from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from typing import Dict, List, Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass(frozen=True)
class Order:
    agent_id: str
    side: Side
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    price: Optional[int] = None
    order_id: Optional[int] = None


@dataclass
class RestingOrder:
    order_id: int
    agent_id: str
    side: Side
    price: int
    quantity: int
    sequence: int


@dataclass(frozen=True)
class Trade:
    step: int
    price: int
    quantity: int
    buyer_id: str
    seller_id: str
    aggressor_side: Side


@dataclass(frozen=True)
class BookSnapshot:
    step: int
    best_bid: Optional[int]
    best_ask: Optional[int]
    mid_price: Optional[float]
    spread: Optional[int]
    bid_depth: int
    ask_depth: int
    imbalance: float
    last_trade_price: Optional[int]


@dataclass
class LimitOrderBook:
    tick_size: int = 1
    bids: List[tuple[int, int, int]] = field(default_factory=list)
    asks: List[tuple[int, int, int]] = field(default_factory=list)
    orders: Dict[int, RestingOrder] = field(default_factory=dict)
    _order_ids: count = field(default_factory=lambda: count(1))
    _sequence: count = field(default_factory=count)
    last_trade_price: Optional[int] = None

    def submit(self, order: Order, step: int) -> List[Trade]:
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        if order.order_type == OrderType.CANCEL:
            if order.order_id is not None:
                self.cancel(order.order_id)
            return []
        if order.order_type == OrderType.LIMIT and order.price is None:
            raise ValueError("Limit orders require a price.")

        trades = self._match(order, step)
        remaining = order.quantity - sum(trade.quantity for trade in trades)
        if remaining > 0 and order.order_type == OrderType.LIMIT:
            self._rest(order, remaining)
        return trades

    def cancel(self, order_id: int) -> bool:
        resting = self.orders.get(order_id)
        if resting is None:
            return False
        resting.quantity = 0
        del self.orders[order_id]
        return True

    def best_bid(self) -> Optional[int]:
        self._clean(self.bids)
        return -self.bids[0][0] if self.bids else None

    def best_ask(self) -> Optional[int]:
        self._clean(self.asks)
        return self.asks[0][0] if self.asks else None

    def depth(self, side: Side, levels: int = 5) -> int:
        prices: set[int] = set()
        total = 0
        heap = self.bids if side == Side.BUY else self.asks
        for signed_price, _, order_id in sorted(heap):
            resting = self.orders.get(order_id)
            if resting is None or resting.quantity <= 0:
                continue
            price = -signed_price if side == Side.BUY else signed_price
            if price not in prices:
                if len(prices) >= levels:
                    break
                prices.add(price)
            total += resting.quantity
        return total

    def snapshot(self, step: int) -> BookSnapshot:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        spread = None
        mid_price = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
        bid_depth = self.depth(Side.BUY)
        ask_depth = self.depth(Side.SELL)
        denom = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / denom if denom else 0.0
        return BookSnapshot(
            step=step,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance=imbalance,
            last_trade_price=self.last_trade_price,
        )

    def random_order_id(self) -> Optional[int]:
        return next(iter(self.orders), None)

    def _match(self, order: Order, step: int) -> List[Trade]:
        trades: List[Trade] = []
        remaining = order.quantity
        opposite = self.asks if order.side == Side.BUY else self.bids

        while remaining > 0:
            self._clean(opposite)
            if not opposite:
                break
            signed_price, _, resting_id = opposite[0]
            resting = self.orders[resting_id]
            resting_price = signed_price if order.side == Side.BUY else -signed_price
            crosses = (
                order.order_type == OrderType.MARKET
                or (order.side == Side.BUY and order.price is not None and order.price >= resting_price)
                or (order.side == Side.SELL and order.price is not None and order.price <= resting_price)
            )
            if not crosses:
                break

            quantity = min(remaining, resting.quantity)
            resting.quantity -= quantity
            remaining -= quantity
            if resting.quantity == 0:
                heappop(opposite)
                self.orders.pop(resting_id, None)

            buyer_id = order.agent_id if order.side == Side.BUY else resting.agent_id
            seller_id = resting.agent_id if order.side == Side.BUY else order.agent_id
            trade = Trade(
                step=step,
                price=resting_price,
                quantity=quantity,
                buyer_id=buyer_id,
                seller_id=seller_id,
                aggressor_side=order.side,
            )
            self.last_trade_price = trade.price
            trades.append(trade)
        return trades

    def _rest(self, order: Order, quantity: int) -> None:
        assert order.price is not None
        order_id = next(self._order_ids)
        sequence = next(self._sequence)
        resting = RestingOrder(
            order_id=order_id,
            agent_id=order.agent_id,
            side=order.side,
            price=order.price,
            quantity=quantity,
            sequence=sequence,
        )
        self.orders[order_id] = resting
        signed_price = -order.price if order.side == Side.BUY else order.price
        heap = self.bids if order.side == Side.BUY else self.asks
        heappush(heap, (signed_price, sequence, order_id))

    def _clean(self, heap: List[tuple[int, int, int]]) -> None:
        while heap and heap[0][2] not in self.orders:
            heappop(heap)

