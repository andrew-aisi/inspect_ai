"""Test records_to_pandas handles mixed types correctly.

This tests the fix for pyarrow conversion failures when columns have
mixed types (e.g., strings and NaN/None values).
"""

import math


def test_records_to_pandas_mixed_string_nan():
    """Test that records with mixed string and NaN values convert correctly.

    This reproduces a bug where columns with both string values and NaN/None
    (which are floats in numpy) caused pyarrow to fail with:
    ArrowTypeError: Expected bytes, got a 'float' object
    """
    from inspect_ai.analysis._dataframe.util import records_to_pandas

    # Simulate records where some samples have a score value and others don't
    records = [
        {"id": "1", "score_value": "correct"},
        {"id": "2", "score_value": "incorrect"},
        {"id": "3", "score_value": float("nan")},  # NaN is a float
        {"id": "4", "score_value": None},
    ]

    # This should not raise an ArrowTypeError
    df = records_to_pandas(records)

    # Verify the dataframe was created correctly
    assert len(df) == 4
    assert "score_value" in df.columns
    assert "id" in df.columns

    # The string values should be preserved
    assert df.loc[0, "score_value"] == "correct"
    assert df.loc[1, "score_value"] == "incorrect"

    # NaN and None should be null/NA in the result
    assert (
        df.loc[2, "score_value"] is None
        or (
            isinstance(df.loc[2, "score_value"], float)
            and math.isnan(df.loc[2, "score_value"])
        )
        or str(df.loc[2, "score_value"]) in ("nan", "<NA>", "None")
    )
    assert df.loc[3, "score_value"] is None or str(df.loc[3, "score_value"]) in (
        "<NA>",
        "None",
    )


def test_records_to_pandas_all_strings():
    """Test that pure string columns still work correctly."""
    from inspect_ai.analysis._dataframe.util import records_to_pandas

    records = [
        {"id": "1", "value": "a"},
        {"id": "2", "value": "b"},
        {"id": "3", "value": "c"},
    ]

    df = records_to_pandas(records)

    assert len(df) == 3
    assert df["value"].tolist() == ["a", "b", "c"]


def test_records_to_pandas_all_none():
    """Test that all-None columns convert correctly."""
    from inspect_ai.analysis._dataframe.util import records_to_pandas

    records = [
        {"id": "1", "value": None},
        {"id": "2", "value": None},
    ]

    df = records_to_pandas(records)

    assert len(df) == 2
    # All None column should not cause issues
    assert df["value"].isna().all()


def test_records_to_pandas_numeric_columns():
    """Test that numeric columns are preserved correctly."""
    from inspect_ai.analysis._dataframe.util import records_to_pandas

    records = [
        {"id": "1", "count": 10, "score": 0.5},
        {"id": "2", "count": 20, "score": 0.8},
    ]

    df = records_to_pandas(records)

    assert len(df) == 2
    # Numeric values should be preserved
    assert df.loc[0, "count"] == 10
    assert df.loc[1, "score"] == 0.8
