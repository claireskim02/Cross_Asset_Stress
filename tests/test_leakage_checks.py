from __future__ import annotations

import pandas as pd

from chronoswan.validation.leakage_checks import audit_feature_matrix, find_suspicious_feature_names


def test_suspicious_feature_names_flag_forward_language() -> None:
    findings = find_suspicious_feature_names(["vix_like", "leaked_future_stress_flag"])

    assert len(findings) == 1
    assert findings[0].column == "leaked_future_stress_flag"


def test_leakage_audit_flags_future_column_and_high_correlation() -> None:
    frame = pd.DataFrame(
        {
            "forecast_timestamp": pd.date_range("2020-01-01", periods=20, tz="UTC"),
            "clean_feature": list(range(20)),
            "leaked_future_stress_flag": [0] * 10 + [1] * 10,
            "target": [0] * 10 + [1] * 10,
        }
    )

    findings = audit_feature_matrix(
        frame,
        target_col="target",
        feature_cols=["clean_feature", "leaked_future_stress_flag"],
    )

    checks = {(finding.check, finding.column) for finding in findings}
    assert ("feature_name", "leaked_future_stress_flag") in checks
    assert ("target_correlation", "leaked_future_stress_flag") in checks

