# Data Dictionary

This dictionary covers the initial synthetic scaffold. Vendor fields should be added only after source metadata is documented.

## Core Timestamps

event_time: Date or period described by the value.

observation_time: Timestamp at which the underlying observation is measured.

release_time: Timestamp at which the value becomes externally released.

ingestion_time: Timestamp at which the value enters the local research system.

earliest_valid_prediction_timestamp: First forecast timestamp allowed to use the value.

vintage: Revision or snapshot identifier.

source: Source identifier.

transformation_window: Lookback or transformation period.

## Synthetic Clean Features

spx_return: Synthetic close-to-close log return.

realized_vol_20: Trailing annualized 20-business-day realized volatility.

downside_semivariance_20: Trailing downside semivariance.

max_drawdown_60: Trailing 60-business-day drawdown from rolling peak.

vix_like: Synthetic option-implied volatility proxy.

skew_like: Synthetic option-skew proxy.

put_call_like: Synthetic put-call demand proxy.

credit_spread_like: Synthetic credit-stress proxy.

futures_basis_like: Synthetic futures-basis proxy.

liquidity_stress_like: Synthetic liquidity-stress proxy.

## Synthetic Invalid Features

leaked_forward_drawdown_20: Computed from future prices. Invalid for forecasting.

leaked_future_stress_flag: Computed from future latent stress state. Invalid for forecasting.

