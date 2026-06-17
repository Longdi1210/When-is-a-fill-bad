from __future__ import annotations

import numpy as np
import pandas as pd


def local_shuffle_trade_signs(events: pd.DataFrame, block_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = events.copy()
    for start in range(0, len(shuffled), block_size):
        idx = shuffled.index[start : start + block_size]
        block = shuffled.loc[idx]
        signed = (block["buy_trade_qty"] - block["sell_trade_qty"]).to_numpy()
        quantities = block["trade_qty"].to_numpy()
        nonzero_signs = np.sign(signed)
        rng.shuffle(nonzero_signs)
        new_buy = np.where(nonzero_signs > 0, quantities, 0.0)
        new_sell = np.where(nonzero_signs < 0, quantities, 0.0)
        shuffled.loc[idx, "buy_trade_qty"] = new_buy
        shuffled.loc[idx, "sell_trade_qty"] = new_sell
        shuffled.loc[idx, "signed_trade_qty"] = new_buy - new_sell
    return shuffled

