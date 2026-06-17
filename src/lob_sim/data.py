from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .agents import MarketState
from .book import LimitOrderBook, Order, OrderType, Side
from .config import ResearchConfig
from .features import compute_features
from .phd_layer import PhDProfile
from .simulation import build_agents, seed_book


def _side_value(side: Side | None) -> str:
    return side.value if side is not None else ""


def _periodic_shock(step: int) -> tuple[float, Side]:
    cycle = (step // 900) % 2
    position = step % 900
    if 120 <= position <= 260:
        middle = 190
        intensity = max(0.0, 1.0 - abs(position - middle) / 70)
        return intensity, Side.BUY if cycle == 0 else Side.SELL
    return 0.0, Side.BUY if cycle == 0 else Side.SELL


def _top_order_id(book: LimitOrderBook, side: Side) -> int | None:
    price = book.best_bid() if side == Side.BUY else book.best_ask()
    if price is None:
        return None
    candidates = [
        order
        for order in book.orders.values()
        if order.side == side and order.price == price and order.quantity > 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda order: order.sequence).order_id


def generate_event_log(config: ResearchConfig, profile: PhDProfile) -> pd.DataFrame:
    rng = random.Random(config.seed)
    book = LimitOrderBook(tick_size=config.tick_size)
    seed_book(book, config.fundamental_price)
    agents = build_agents(profile)
    rows: list[dict] = []
    last_side = Side.BUY
    latent_directional_drift = 0.0

    for step in range(1, config.steps + 1):
        before = book.snapshot(step)
        profile_shock = profile.shock_intensity(step)
        periodic_shock, periodic_side = _periodic_shock(step)
        shock = max(profile_shock, periodic_shock)
        direction = 1 if periodic_side == Side.BUY else -1
        drifting_fundamental = config.fundamental_price + int(direction * 12 * shock)
        state = MarketState(step=step, fundamental_price=drifting_fundamental, last_side=last_side)
        agent = rng.choice(agents)
        top_cancel_side = rng.choice([Side.BUY, Side.SELL])
        top_cancel_id = _top_order_id(book, top_cancel_side)
        if periodic_shock < 0.1 and top_cancel_id is not None and rng.random() < 0.05:
            order = Order("synthetic_top_cancel", top_cancel_side, 1, OrderType.CANCEL, order_id=top_cancel_id)
        else:
            order = agent.decide(book, state, profile, rng)
        if periodic_shock >= 0.25 and rng.random() < 0.65:
            order = Order(
                agent_id=f"synthetic_informed_{step}",
                side=periodic_side,
                quantity=rng.randint(25, 90),
                order_type=OrderType.MARKET,
            )
            latent_directional_drift += (1 if periodic_side == Side.BUY else -1) * rng.uniform(0.4, 1.6) * periodic_shock
        elif order.agent_id.startswith("informed_") and shock >= 0.5:
            shock_cycle = (step - profile.shock_start) // max(1, (profile.shock_end - profile.shock_start) // 4)
            directional_side = Side.BUY if shock_cycle % 2 == 0 else Side.SELL
            order = Order(
                agent_id=order.agent_id,
                side=directional_side,
                quantity=order.quantity,
                order_type=order.order_type,
                price=order.price,
                order_id=order.order_id,
            )
        last_side = order.side

        cancel_side = None
        cancel_qty = 0
        cancel_price = None
        if order.order_type == OrderType.CANCEL and order.order_id in book.orders:
            resting = book.orders[order.order_id]
            cancel_side = resting.side
            cancel_qty = resting.quantity
            cancel_price = resting.price

        trades = book.submit(order, step)
        after = book.snapshot(step)
        latent_directional_drift *= 0.985
        markout_mid_price = (after.mid_price if after.mid_price is not None else config.fundamental_price) + latent_directional_drift
        features = compute_features(after, trades)
        buy_trade_qty = sum(trade.quantity for trade in trades if trade.aggressor_side == Side.BUY)
        sell_trade_qty = sum(trade.quantity for trade in trades if trade.aggressor_side == Side.SELL)
        trade_price = trades[-1].price if trades else None

        row = {
            "step": step,
            "event_type": order.order_type.value,
            "order_side": order.side.value,
            "order_price": order.price if order.price is not None else np.nan,
            "order_qty": order.quantity,
            "cancel_side": _side_value(cancel_side),
            "cancel_qty": cancel_qty,
            "cancel_price": cancel_price if cancel_price is not None else np.nan,
            "trade_price": trade_price if trade_price is not None else np.nan,
            "buy_trade_qty": buy_trade_qty,
            "sell_trade_qty": sell_trade_qty,
            "trade_qty": buy_trade_qty + sell_trade_qty,
            "best_bid": after.best_bid,
            "best_ask": after.best_ask,
            "mid_price": after.mid_price,
            "markout_mid_price": markout_mid_price,
            "spread": after.spread,
            "bid_depth": after.bid_depth,
            "ask_depth": after.ask_depth,
            "imbalance": after.imbalance,
            "last_trade_price": after.last_trade_price,
            "microprice": features.microprice,
            "microprice_deviation": features.microprice_deviation,
            "spread_bps": features.spread_bps,
            "queue_imbalance": features.queue_imbalance,
            "signed_trade_volume": features.signed_trade_volume,
            "order_flow_imbalance": features.order_flow_imbalance,
            "pre_best_bid": before.best_bid,
            "pre_best_ask": before.best_ask,
            "pre_bid_depth": before.bid_depth,
            "pre_ask_depth": before.ask_depth,
        }
        rows.append(row)

    events = pd.DataFrame(rows)
    events["signed_trade_qty"] = events["buy_trade_qty"] - events["sell_trade_qty"]
    window = config.rolling_window
    events["recent_signed_trade_flow"] = events["signed_trade_qty"].rolling(window, min_periods=1).sum()
    events["recent_trade_volume"] = events["trade_qty"].rolling(window, min_periods=1).sum()
    events["recent_volatility"] = events["mid_price"].diff().abs().rolling(window, min_periods=1).mean().fillna(0.0)
    events["recent_mid_move"] = events["mid_price"].diff(window).fillna(0.0)
    return events
