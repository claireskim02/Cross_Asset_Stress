# Leakage Protocol

Leakage is a primary research object in ChronoSwan, not a footnote.

## Point-In-Time Rule

For every forecast timestamp, the model may use only records whose earliest valid prediction timestamp is less than or equal to the forecast timestamp.

Every feature record must include:

- event_time;
- observation_time;
- release_time;
- ingestion_time;
- vintage;
- source;
- transformation_window;
- earliest_valid_prediction_timestamp.

Documents must include:

- original publication timestamp;
- time zone;
- first-seen timestamp;
- source identifier;
- document hash;
- retrieval timestamp;
- revision status;
- earliest permissible prediction timestamp.

## Data Leakage Checks

The first scaffold checks:

- feature names containing future, forward, leaked, target, label, outcome, or post-event;
- availability timestamps later than forecast timestamps;
- near-perfect target correlation;
- known synthetic leakage tags.

The synthetic data intentionally includes invalid features so the audit has something to catch.

## Validation Leakage Controls

Do not use random train-test splits for valid claims.

For a 20-day forward label, remove training rows whose forward outcome windows overlap a validation interval. Add an embargo after validation windows to reduce spillover from adjacent observations.

All preprocessing, feature selection, scaling, calibration, hyperparameter tuning, and threshold selection must be fit inside each training fold.

## LLM Leakage Controls

Prompt instructions are necessary but insufficient.

Valid LLM experiments require:

- retrieval restricted by document availability timestamps;
- full system and user prompt logging;
- context document IDs and hashes;
- removal of event names and post-event commentary in sanitized experiments;
- placebo and shuffled-context experiments;
- comparison of pre-event and post-event documents;
- explicit contaminated-agent condition;
- contamination-probe tasks;
- separate reporting of retrieval leakage and parametric-memory leakage.

An unrestricted historical LLM result is not evidence of predictive ability.

