from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .book import BookSnapshot, Side, Trade


@dataclass(frozen=True)
class MicrostructureFeatures:
    step: int
    mid_price: Optional[float]
    microprice: Optional[float]
    microprice_deviation: float
    spread_bps: Optional[float]
    queue_imbalance: float
    signed_trade_volume: int
    order_flow_imbalance: float

    def to_row(self) -> dict:
        return asdict(self)


def compute_features(snapshot: BookSnapshot, trades: list[Trade]) -> MicrostructureFeatures:
    total_depth = snapshot.bid_depth + snapshot.ask_depth
    queue_imbalance = (snapshot.bid_depth - snapshot.ask_depth) / total_depth if total_depth else 0.0
    microprice = None
    microprice_deviation = 0.0
    if snapshot.best_bid is not None and snapshot.best_ask is not None and total_depth:
        microprice = (
            snapshot.best_ask * snapshot.bid_depth + snapshot.best_bid * snapshot.ask_depth
        ) / total_depth
        if snapshot.mid_price is not None:
            microprice_deviation = microprice - snapshot.mid_price

    spread_bps = None
    if snapshot.spread is not None and snapshot.mid_price:
        spread_bps = 10_000 * snapshot.spread / snapshot.mid_price

    signed_trade_volume = sum(
        trade.quantity if trade.aggressor_side == Side.BUY else -trade.quantity
        for trade in trades
    )
    absolute_trade_volume = sum(trade.quantity for trade in trades)
    order_flow_imbalance = signed_trade_volume / absolute_trade_volume if absolute_trade_volume else 0.0

    return MicrostructureFeatures(
        step=snapshot.step,
        mid_price=snapshot.mid_price,
        microprice=microprice,
        microprice_deviation=microprice_deviation,
        spread_bps=spread_bps,
        queue_imbalance=queue_imbalance,
        signed_trade_volume=signed_trade_volume,
        order_flow_imbalance=order_flow_imbalance,
    )

