When Is a Fill Bad?

1. Research Question

Can pre-fill order-book states explain when a passive buy fill preserves spread capture and when it suffers adverse post-fill price drift?

2. Model

The project uses a minimal event-driven simulator for a passive buy order resting at the best bid.

Each simulation is defined by three initial state variables:

* queue volume ahead of the order;
* order-book imbalance;
* short-horizon volatility.

The passive order fills when market-sell volume and cancellations consume the queue ahead of it.

Order-book imbalance is defined as

[
I_t =
\frac{D_t^{bid}-D_t^{ask}}
{D_t^{bid}+D_t^{ask}}.
]

Negative imbalance represents a relatively weak bid side. In the controlled model, negative imbalance increases sell-side queue depletion and creates negative expected short-horizon mid-price drift.

After a fill, execution quality is evaluated over a fixed markout horizon.

For a passive buy order,

[
\Delta m_{t,\tau}

m_{t+\tau}-m_t^{fill}.
]

The order initially captures half the spread,

[
SC_t = m_t^{fill}-p_t^{fill}.
]

Net execution value is defined as

[
NEV_{t,\tau}

SC_t+\Delta m_{t,\tau}.
]

A fill is classified as bad when

[
NEV_{t,\tau}<0.
]

3. Interpretation Boundary

The simulator deliberately imposes a short-horizon relationship between imbalance, sell pressure, and price drift.

The current figures therefore test whether the measurement pipeline can recover a controlled mechanism. They do not establish an empirical market relationship.

The next research stage must challenge this mechanism using richer state-dependent order flow and real limit-order-book data.
