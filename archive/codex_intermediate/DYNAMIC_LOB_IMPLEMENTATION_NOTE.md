# Dynamic LOB Implementation Note

Existing baseline retained: static P1/P2/P3 proxy validation remains available.

Strict temporal design added: shock formation [t-10s,t], early absorption (t,t+5s], and future outcomes (t+5s,t+H]. Absorption excludes quote-survival and markout outcomes.

Dynamic objects added: multi-level depth, shock ratios, potential penetration classes, early post-shock absorption, conditional best-quote survival, future markout response paths, local projections, and expanded stratified shock null.

Data-resolution limitation: one-second aggregates support potential penetration and best-quote survival proxies, not exact FIFO fills or intrasecond execution paths.
