from pathlib import Path

from src.churn_pipeline import EXCLUDED_FEATURES, load_source_data, prepare_dataset


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_prepared_dataset_has_expected_target_and_no_leakage() -> None:
    customers, population = load_source_data(DATA_DIR)
    X, y, frame = prepare_dataset(customers, population)

    assert len(customers) == 7_043
    assert len(frame) == 6_589
    assert int(y.sum()) == 1_869
    assert set(y.unique()) == {0, 1}
    assert "Population" in X.columns
    assert not set(EXCLUDED_FEATURES).intersection(X.columns)


def test_zip_population_merge_does_not_duplicate_customers() -> None:
    customers, population = load_source_data(DATA_DIR)
    _, _, frame = prepare_dataset(customers, population)

    expected = customers["Customer Status"].isin(["Stayed", "Churned"]).sum()
    assert len(frame) == expected
    assert frame["Population"].notna().all()

