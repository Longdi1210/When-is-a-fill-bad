from __future__ import annotations

import random
from dataclasses import dataclass

from .book import LimitOrderBook, Order, OrderType, Side
from .phd_layer import PhDProfile


@dataclass
class MarketState:
    step: int
    fundamental_price: int
    last_side: Side


class Agent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def decide(self, book: LimitOrderBook, state: MarketState, profile: PhDProfile, rng: random.Random) -> Order:
        raise NotImplementedError


class NoiseTrader(Agent):
    def decide(self, book: LimitOrderBook, state: MarketState, profile: PhDProfile, rng: random.Random) -> Order:
        side = state.last_side if rng.random() < profile.order_flow_autocorrelation else rng.choice([Side.BUY, Side.SELL])
        best_bid = book.best_bid() or state.fundamental_price - 1
        best_ask = book.best_ask() or state.fundamental_price + 1
        price = best_bid - rng.randint(0, 3) if side == Side.BUY else best_ask + rng.randint(0, 3)
        return Order(self.agent_id, side, rng.randint(1, 8), OrderType.LIMIT, price)


class InformedTrader(Agent):
    def decide(self, book: LimitOrderBook, state: MarketState, profile: PhDProfile, rng: random.Random) -> Order:
        intensity = profile.shock_intensity(state.step)
        side = Side.BUY if intensity >= 0.5 else rng.choice([Side.BUY, Side.SELL])
        quantity = rng.randint(4, 12) + int(18 * intensity)
        if rng.random() < 0.65 + 0.25 * intensity:
            return Order(self.agent_id, side, quantity, OrderType.MARKET)
        anchor = state.fundamental_price + int(4 * intensity)
        price = anchor + rng.randint(0, 4) if side == Side.BUY else anchor - rng.randint(0, 4)
        return Order(self.agent_id, side, quantity, OrderType.LIMIT, price)


class ResilienceLiquidityProvider(Agent):
    def decide(self, book: LimitOrderBook, state: MarketState, profile: PhDProfile, rng: random.Random) -> Order:
        snapshot = book.snapshot(state.step)
        side = Side.SELL if snapshot.imbalance > 0 else Side.BUY
        best_bid = book.best_bid() or state.fundamental_price - 1
        best_ask = book.best_ask() or state.fundamental_price + 1
        inside = profile.shock_intensity(state.step) > 0.3
        if side == Side.BUY:
            price = best_bid + (1 if inside and best_bid + 1 < best_ask else 0)
        else:
            price = best_ask - (1 if inside and best_ask - 1 > best_bid else 0)
        quantity = rng.randint(6, 16)
        return Order(self.agent_id, side, quantity, OrderType.LIMIT, price)


class CancellationAgent(Agent):
    def decide(self, book: LimitOrderBook, state: MarketState, profile: PhDProfile, rng: random.Random) -> Order:
        order_id = book.random_order_id()
        if order_id is not None and rng.random() < profile.cancellation_pressure + 0.12 * profile.shock_intensity(state.step):
            return Order(self.agent_id, rng.choice([Side.BUY, Side.SELL]), 1, OrderType.CANCEL, order_id=order_id)
        return NoiseTrader(self.agent_id).decide(book, state, profile, rng)

