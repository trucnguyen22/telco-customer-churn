"""Tests for the raw-data loading boundary (telco_churn.data)."""

import pytest

from telco_churn.data import RAW_COLUMNS, load_raw


def test_load_raw_accepts_a_valid_extract(tmp_path, make_raw):
    csv = tmp_path / "extract.csv"
    make_raw().to_csv(csv, index=False)

    df = load_raw(csv)

    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 1


def test_missing_column_raises_and_names_it(tmp_path, make_raw):
    csv = tmp_path / "broken.csv"
    make_raw().drop(columns=["TotalCharges"]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="TotalCharges"):
        load_raw(csv)


def test_unexpected_column_raises_and_names_it(tmp_path, make_raw):
    csv = tmp_path / "broken.csv"
    df = make_raw()
    df["LoyaltyScore"] = 7
    df.to_csv(csv, index=False)

    with pytest.raises(ValueError, match="LoyaltyScore"):
        load_raw(csv)
