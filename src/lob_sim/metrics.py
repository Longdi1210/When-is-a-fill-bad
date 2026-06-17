from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable, List

from .book import BookSnapshot, Trade
from .features import MicrostructureFeatures
from .phd_layer import PhDProfile


def realized_volatility(snapshots: Iterable[BookSnapshot]) -> float:
    mids = [snap.mid_price for snap in snapshots if snap.mid_price is not None and snap.mid_price > 0]
    if len(mids) < 2:
        return 0.0
    returns = [math.log(curr / prev) for prev, curr in zip(mids, mids[1:]) if prev > 0 and curr > 0]
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance)


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def resilience_score(snapshots: List[BookSnapshot], profile: PhDProfile) -> float:
    pre = [snap.spread for snap in snapshots if snap.spread is not None and snap.step < profile.shock_start]
    post = [snap.spread for snap in snapshots if snap.spread is not None and snap.step > profile.shock_end]
    if not pre or not post:
        return 0.0
    baseline = average([float(value) for value in pre])
    recovery = average([float(value) for value in post[: max(5, len(post) // 3)]])
    if baseline <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(recovery - baseline) / baseline)


def summarize(
    snapshots: List[BookSnapshot],
    trades: List[Trade],
    profile: PhDProfile,
    features: List[MicrostructureFeatures] | None = None,
) -> dict:
    spreads = [float(snap.spread) for snap in snapshots if snap.spread is not None]
    imbalances = [abs(snap.imbalance) for snap in snapshots]
    microprice_deviations = [abs(row.microprice_deviation) for row in features or []]
    ofi_values = [abs(row.order_flow_imbalance) for row in features or []]
    volume = sum(trade.quantity for trade in trades)
    return {
        "profile": {
            "title": profile.title,
            "research_question": profile.research_question,
            "hypothesis": profile.hypothesis,
            "research_features": profile.research_features,
        },
        "steps": len(snapshots),
        "trades": len(trades),
        "trade_volume": volume,
        "average_spread": average(spreads),
        "average_abs_imbalance": average(imbalances),
        "average_abs_microprice_deviation": average(microprice_deviations),
        "average_abs_order_flow_imbalance": average(ofi_values),
        "realized_volatility": realized_volatility(snapshots),
        "resilience_score": resilience_score(snapshots, profile),
        "last_snapshot": asdict(snapshots[-1]) if snapshots else None,
    }
