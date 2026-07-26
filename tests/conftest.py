"""Shared fixtures for the telco_churn test suite.

pytest automatically discovers this file and makes its fixtures available
to every test module in this directory -- no imports needed.
"""

import pandas as pd
import pytest

# One deliberately hand-checkable customer. Values are chosen so expected
# results are mental arithmetic: TotalCharges 380 over tenure 4 gives a
# lifetime average of 95, so charges_delta = 100 - 95 = 5.0.
# TotalCharges is a *string* because that is how the raw extract arrives.
# Key order matches data.RAW_COLUMNS so CSVs written from this row have
# the exact raw column order.
_BASE_CUSTOMER = {
    "customerID": "0001-TEST",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 4,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 100.0,
    "TotalCharges": "380",
    "Churn": "No",
}


@pytest.fixture
def make_raw():
    """Factory fixture: build a one-row raw extract, overriding per test.

    Usage: ``make_raw(tenure=0, TotalCharges=" ")`` returns a DataFrame
    identical to the base customer except for the overridden fields.
    """

    def _make(**overrides: object) -> pd.DataFrame:
        return pd.DataFrame([{**_BASE_CUSTOMER, **overrides}])

    return _make
