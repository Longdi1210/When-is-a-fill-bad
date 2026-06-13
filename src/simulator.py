from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
import pandas as pd


@dataclass
class LOBState:
    mid_price: float
    spread: float
    bid_depth: float
    ask_depth: float
    volatility: float

    @property
    def best_bid(self) -> float:
        return self.mid_price - self.spread / 2

    @property
    def best_ask(self) -> float:
        return self.mid_price + self.spread / 2

    @property
    def imbalance(self) -> float:
        denom = self.bid_depth + self.ask_depth
        if denom <= 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / denom


@dataclass
class PassiveOrder:
    side: str
    size: float
    queue_ahead: float
    placed_price: float
    placed_mid: float
    placed_spread: float
    placed_imbalance: float
    placed_volatility: float
    is_filled: bool = False
    fill_time: Optional[int] = None
    fill_price: Optional[float] = None


@dataclass
class SimulationConfig:
    n_steps: int = 200
    post_fill_horizon: int = 20

    initial_mid: float = 100.0
    spread: float = 0.02
    initial_bid_depth: float = 100.0
    initial_ask_depth: float = 100.0

    volatility: float = 0.01
    market_sell_rate: float = 0.4
    cancellation_rate: float = 0.2
    limit_replenish_rate: float = 0.2

    order_size: float = 1.0
    initial_queue_ahead: float = 20.0

    imbalance_price_impact: float = 0.002
    seed: Optional[int] = None


class SimpleLOBSimulator:
    """
    v0.1 simplified LOB simulator for passive buy execution quality.

    Goal:
    pre-fill state -> queue evolution -> fill event -> post-fill drift -> execution metrics.

    This is a mechanism-study simulator, not a production market simulator.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.state = LOBState(
            mid_price=config.initial_mid,
            spread=config.spread,
            bid_depth=config.initial_bid_depth,
            ask_depth=config.initial_ask_depth,
            volatility=config.volatility,
        )

        self.order = PassiveOrder(
            side="buy",
            size=config.order_size,
            queue_ahead=config.initial_queue_ahead,
            placed_price=self.state.best_bid,
            placed_mid=self.state.mid_price,
            placed_spread=self.state.spread,
            placed_imbalance=self.state.imbalance,
            placed_volatility=self.state.volatility,
        )

        self.history: List[Dict] = []

    def step(self, t: int) -> None:
        """
        One event-time step.
        """

        # 1. Market sell volume consumes bid-side queue.
        market_sell_volume = self.rng.poisson(self.config.market_sell_rate)

        # 2. Bid-side cancellations remove queue ahead with some probability.
        cancellation_volume = self.rng.poisson(self.config.cancellation_rate)

        # 3. Limit order replenishment changes bid/ask depth.
        bid_replenish = self.rng.poisson(self.config.limit_replenish_rate)
        ask_replenish = self.rng.poisson(self.config.limit_replenish_rate)

        # Update depth.
        self.state.bid_depth = max(
            0.0,
            self.state.bid_depth - market_sell_volume - cancellation_volume + bid_replenish,
        )
        self.state.ask_depth = max(
            0.0,
            self.state.ask_depth + ask_replenish,
        )

        # Update queue ahead for passive buy.
        if not self.order.is_filled:
            self.order.queue_ahead = max(
                0.0,
                self.order.queue_ahead - market_sell_volume - cancellation_volume,
            )

            if self.order.queue_ahead <= 0.0:
                self.order.is_filled = True
                self.order.fill_time = t
                self.order.fill_price = self.order.placed_price

        # 4. Mid-price evolution.
        # Negative imbalance means ask depth > bid depth; selling pressure can push mid down.
        imbalance_term = self.config.imbalance_price_impact * self.state.imbalance
        random_shock = self.rng.normal(0.0, self.state.volatility)
        self.state.mid_price += imbalance_term + random_shock

        # Record.
        self.history.append({
            "t": t,
            "mid_price": self.state.mid_price,
            "best_bid": self.state.best_bid,
            "best_ask": self.state.best_ask,
            "spread": self.state.spread,
            "bid_depth": self.state.bid_depth,
            "ask_depth": self.state.ask_depth,
            "imbalance": self.state.imbalance,
            "queue_ahead": self.order.queue_ahead,
            "is_filled": self.order.is_filled,
            "fill_time": self.order.fill_time,
            "fill_price": self.order.fill_price,
        })

    def run(self) -> pd.DataFrame:
        for t in range(self.config.n_steps):
            self.step(t)

        return pd.DataFrame(self.history)

    def summarize(self) -> Dict:
        df = pd.DataFrame(self.history)

        filled = self.order.is_filled

        result = {
            "filled": filled,
            "placed_price": self.order.placed_price,
            "placed_mid": self.order.placed_mid,
            "placed_spread": self.order.placed_spread,
            "placed_imbalance": self.order.placed_imbalance,
            "placed_volatility": self.order.placed_volatility,
            "initial_queue_ahead": self.config.initial_queue_ahead,
            "fill_time": self.order.fill_time,
            "fill_price": self.order.fill_price,
        }

        if filled:
            fill_time = self.order.fill_time
            horizon_time = min(
                fill_time + self.config.post_fill_horizon,
                len(df) - 1
            )

            post_mid = df.loc[horizon_time, "mid_price"]
            drift = post_mid - self.order.fill_price

            spread_capture = self.order.placed_spread / 2
            adverse_drift = max(0.0, -drift)
            net_execution_value = spread_capture - adverse_drift

            result.update({
                "post_mid": post_mid,
                "post_fill_drift": drift,
                "spread_capture": spread_capture,
                "adverse_drift": adverse_drift,
                "net_execution_value": net_execution_value,
            })
        else:
            result.update({
                "post_mid": np.nan,
                "post_fill_drift": np.nan,
                "spread_capture": self.order.placed_spread / 2,
                "adverse_drift": np.nan,
                "net_execution_value": np.nan,
            })

        return result


def run_single_simulation(seed: int = 0) -> Dict:
    config = SimulationConfig(seed=seed)
    sim = SimpleLOBSimulator(config)
    sim.run()
    return sim.summarize()