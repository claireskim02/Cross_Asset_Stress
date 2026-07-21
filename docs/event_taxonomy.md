# Event Taxonomy

ChronoSwan separates event definitions from model selection.

## Event Families

Forward drawdown: Minimum forward index return over a configured horizon breaches a threshold.

Volatility regime: Forward realized volatility enters a configured historical upper-tail regime.

VIX threshold: VIX or a VIX-like feature crosses a configured threshold within the horizon.

Joint stress: Multiple dimensions, such as drawdown, volatility, credit, and liquidity, deteriorate together.

Time to event: Business days until the next event onset.

## Event States

Onset: First row of a contiguous event window.

Continuation: Later row inside the same event window.

Recovery: First non-event row after an event window.

Calm: Non-event row not immediately following an event.

Insufficient forward window: Row without enough future observations to compute the label.

## Event-Level Evaluation

Daily-row metrics can overstate performance during long crises. Event-level reporting should count whether a model detected an event before or near onset, not whether it correctly classified many adjacent crisis rows.

