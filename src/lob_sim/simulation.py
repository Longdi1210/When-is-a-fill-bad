from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

from .agents import CancellationAgent, InformedTrader, MarketState, NoiseTrader, ResilienceLiquidityProvider
from .book import LimitOrderBook, Order, OrderType, Side, Trade
from .features import compute_features
from .metrics import summarize
from .phd_layer import PhDProfile


def seed_book(book: LimitOrderBook, fundamental_price: int) -> None:
    for level in range(1, 7):
        quantity = 10 + level * 3
        book.submit(Order("seed", Side.BUY, quantity, OrderType.LIMIT, fundamental_price - level), step=0)
        book.submit(Order("seed", Side.SELL, quantity, OrderType.LIMIT, fundamental_price + level), step=0)


def build_agents(profile: PhDProfile, population: int = 100):
    informed_n = max(1, int(population * profile.informed_agent_share))
    resilience_n = max(1, int(population * profile.resilience_agent_share))
    cancellation_n = max(1, int(population * 0.10))
    noise_n = max(1, population - informed_n - resilience_n - cancellation_n)
    agents = []
    agents.extend(InformedTrader(f"informed_{i}") for i in range(informed_n))
    agents.extend(ResilienceLiquidityProvider(f"resilience_{i}") for i in range(resilience_n))
    agents.extend(CancellationAgent(f"cancel_{i}") for i in range(cancellation_n))
    agents.extend(NoiseTrader(f"noise_{i}") for i in range(noise_n))
    return agents


def run_simulation(profile: PhDProfile, steps: int = 500, seed: int = 42, fundamental_price: int = 10_000) -> dict:
    rng = random.Random(seed)
    book = LimitOrderBook()
    seed_book(book, fundamental_price)
    agents = build_agents(profile)
    snapshots = []
    feature_rows = []
    trades: list[Trade] = []
    last_side = Side.BUY

    for step in range(1, steps + 1):
        shock = profile.shock_intensity(step)
        drifting_fundamental = fundamental_price + int(12 * shock)
        state = MarketState(step=step, fundamental_price=drifting_fundamental, last_side=last_side)
        agent = rng.choice(agents)
        order = agent.decide(book, state, profile, rng)
        last_side = order.side
        step_trades = book.submit(order, step)
        trades.extend(step_trades)
        snapshot = book.snapshot(step)
        snapshots.append(snapshot)
        feature_rows.append(compute_features(snapshot, step_trades))

    return {
        "snapshots": snapshots,
        "trades": trades,
        "features": feature_rows,
        "summary": summarize(snapshots, trades, profile, feature_rows),
    }


def write_outputs(result: dict, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snapshots = result["snapshots"]
    with (output_path / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(snapshots[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(snapshot) for snapshot in snapshots)

    trades = result["trades"]
    with (output_path / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["step", "price", "quantity", "buyer_id", "seller_id", "aggressor_side"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            row = asdict(trade)
            row["aggressor_side"] = trade.aggressor_side.value
            writer.writerow(row)

    features = result["features"]
    with (output_path / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0].to_row().keys()))
        writer.writeheader()
        writer.writerows(feature.to_row() for feature in features)

    (output_path / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
