import unittest
import numpy as np
import pandas as pd
from modules.intelligent_cleaner import adaptive_clean, profile_dataset
from modules.cleaning_evaluator import compare_quality

class IntelligentCleanerTests(unittest.TestCase):
    def test_profile_counts_missing(self):
        df = pd.DataFrame({'age': [20, np.nan, 30], 'city': [' Salem ', 'Chennai', 'Chennai']})
        self.assertEqual(profile_dataset(df)['missing_cells'], 1)

    def test_imputation_and_duplicates(self):
        df = pd.DataFrame({'age': [20.0, np.nan, 30.0, 30.0], 'city': [' SALEM ', 'Chennai', 'Chennai', 'Chennai']})
        cleaned, summary = adaptive_clean(df)
        self.assertEqual(len(cleaned), 3)
        self.assertFalse(cleaned['age'].isna().any())
        self.assertEqual(summary['duplicate_rows_removed'], 1)

    def test_anomalies_are_flagged_not_deleted(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({'x': np.r_[rng.normal(size=100), 20.0], 'y': np.r_[rng.normal(size=100), 20.0]})
        cleaned, summary = adaptive_clean(df)
        self.assertEqual(len(cleaned), len(df))
        self.assertIn('_auto_clean_anomaly', cleaned.columns)
        self.assertGreaterEqual(summary['anomaly_rows_flagged'], 0)


    def test_quality_components_are_reported(self):
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
        cleaned, summary = adaptive_clean(df)
        self.assertIn("quality_components_before", summary)
        self.assertIn("completeness_pct", summary["quality_components_before"])
        self.assertEqual(summary["quality_components_after"]["completeness_pct"], 100.0)

    def test_anomaly_audit_metadata_is_reported(self):
        rng = np.random.default_rng(12)
        df = pd.DataFrame({
            "first": np.r_[rng.normal(size=80), 18.0],
            "second": np.r_[rng.normal(size=80), 18.0],
        })
        cleaned, summary = adaptive_clean(df)
        self.assertIn("anomaly_percentage", summary)
        self.assertEqual(summary["anomaly_method"], "Isolation Forest")
        self.assertEqual(len(cleaned), len(df))


    def test_numeric_imputation_is_benchmarked(self):
        rng = np.random.default_rng(21)
        feature = rng.normal(40, 8, 140)
        related = feature * 1.8 + rng.normal(0, 1, 140)
        df = pd.DataFrame({"feature": feature, "related": related})
        df.loc[rng.choice(len(df), 25, replace=False), "feature"] = np.nan

        cleaned, summary = adaptive_clean(df)
        decision = next(item for item in summary["decisions"] if item["column"] == "feature")

        self.assertEqual(decision["benchmark"]["status"], "completed")
        self.assertIn("mean", decision["benchmark"]["metrics"])
        self.assertIn("median", decision["benchmark"]["metrics"])
        self.assertIn("knn", decision["benchmark"]["metrics"])
        self.assertFalse(cleaned["feature"].isna().any())


    def test_identifiers_and_target_columns_are_excluded_from_anomaly_features(self):
        rng = np.random.default_rng(33)
        df = pd.DataFrame({
            "PassengerId": np.arange(1, 101),
            "Survived": rng.integers(0, 2, 100),
            "Pclass": rng.integers(1, 4, 100),
            "SibSp": rng.integers(0, 4, 100),
            "Age": rng.normal(35, 12, 100),
            "Fare": rng.lognormal(3, 0.7, 100),
        })
        _, summary = adaptive_clean(df)
        used = summary["anomaly_features_used"]
        excluded = summary["anomaly_features_excluded"]
        self.assertNotIn("PassengerId", used)
        self.assertNotIn("Survived", used)
        self.assertNotIn("SibSp", used)
        self.assertIn("PassengerId", excluded)
        self.assertIn("Survived", excluded)

    def test_imputation_benchmark_uses_repeated_masks_without_target_leakage(self):
        rng = np.random.default_rng(44)
        age = rng.normal(35, 10, 160)
        df = pd.DataFrame({
            "PassengerId": np.arange(1, 161),
            "Survived": rng.integers(0, 2, 160),
            "Age": age,
            "Fare": age * 2 + rng.normal(0, 5, 160),
        })
        df.loc[rng.choice(len(df), 25, replace=False), "Age"] = np.nan
        _, summary = adaptive_clean(df)
        age_decision = next(item for item in summary["decisions"] if item["column"] == "Age")
        benchmark = age_decision["benchmark"]
        self.assertEqual(benchmark["repeats"], 5)
        self.assertNotIn("PassengerId", benchmark["predictor_features"])
        self.assertNotIn("Survived", benchmark["predictor_features"])
        self.assertIn("Fare", benchmark["predictor_features"])


    def test_contextual_anomaly_grouping_is_detected_and_user_overridable(self):
        rng = np.random.default_rng(66)
        df = pd.DataFrame({
            "PassengerId": np.arange(1, 181),
            "Survived": rng.integers(0, 2, 180),
            "Pclass": np.repeat([1, 2, 3], 60),
            "Age": rng.normal(35, 10, 180),
            "Fare": np.r_[rng.normal(100, 10, 60), rng.normal(20, 5, 60), rng.normal(10, 3, 60)],
        })
        _, automatic = adaptive_clean(df)
        self.assertEqual(automatic["anomaly_grouping_mode"], "automatic")
        self.assertEqual(automatic["anomaly_group_columns"], ["Pclass"])

        _, selected = adaptive_clean(df, group_columns=["Pclass"])
        self.assertEqual(selected["anomaly_grouping_mode"], "user-selected")
        self.assertEqual(selected["anomaly_group_columns"], ["Pclass"])

    def test_user_selected_target_is_excluded_from_feature_sets(self):
        rng = np.random.default_rng(55)
        df = pd.DataFrame({
            "measure_a": rng.normal(10, 2, 120),
            "measure_b": rng.normal(20, 3, 120),
            "outcome_score": rng.normal(50, 8, 120),
        })
        df.loc[rng.choice(len(df), 20, replace=False), "measure_a"] = np.nan
        _, summary = adaptive_clean(df, exclude_columns=["outcome_score"])
        decision = next(item for item in summary["decisions"] if item["column"] == "measure_a")
        self.assertNotIn("outcome_score", decision["benchmark"]["predictor_features"])
        self.assertIn("outcome_score", decision["benchmark"]["excluded_predictor_columns"])
        self.assertIn("outcome_score", summary["selected_exclusion_columns"])


    def test_placeholder_tokens_are_counted_in_initial_quality_profile(self):
        df = pd.DataFrame({"workclass": ["Private", "?", "N/A", "State-gov"]})
        profile = profile_dataset(df)
        self.assertEqual(profile["missing_cells"], 2)
        self.assertEqual(profile["columns"][0]["missing"], 2)

    def test_imputation_comparison_includes_five_methods(self):
        rng = np.random.default_rng(101)
        x = rng.normal(40, 8, 120)
        y = x * 1.5 + rng.normal(0, 2, 120)
        df = pd.DataFrame({"x": x, "y": y})
        df.loc[rng.choice(len(df), 20, replace=False), "x"] = np.nan
        cleaned, summary = adaptive_clean(df)
        decision = next(item for item in summary["decisions"] if item["column"] == "x")
        self.assertEqual(decision["benchmark"]["status"], "completed")
        self.assertEqual(set(decision["benchmark"]["metrics"]), {"mean", "median", "knn", "missforest", "mice"})
        self.assertIn(decision["selected_method"], decision["benchmark"]["metrics"])
        self.assertFalse(cleaned["x"].isna().any())

    def test_anomaly_comparison_reports_multiple_detectors_and_reasons(self):
        rng = np.random.default_rng(202)
        df = pd.DataFrame({
            "measurement_a": np.r_[rng.normal(10, 1, 100), 30.0],
            "measurement_b": np.r_[rng.normal(20, 2, 100), 70.0],
        })
        cleaned, summary = adaptive_clean(df)
        self.assertIn(summary["anomaly_selected_method"], {"isolation_forest", "local_outlier_factor", "dbscan", "one_class_svm"})
        self.assertGreaterEqual(len(summary["anomaly_algorithm_comparison"]), 2)
        self.assertIn("_auto_clean_anomaly_reason", cleaned.columns)
        self.assertIn("anomaly_algorithm_selection_note", summary)
        if summary["anomaly_rows_flagged"]:
            flagged = cleaned.loc[cleaned["_auto_clean_anomaly"]]
            self.assertTrue(flagged["_auto_clean_anomaly_reason"].str.len().gt(0).all())
            self.assertIn("_auto_clean_anomaly_reason", summary["anomaly_review_preview"][0])

    def test_quality_comparison_counts_placeholder_tokens(self):
        before = pd.DataFrame({"category": ["A", "?", "N/A"]})
        after = pd.DataFrame({"category": ["A", "Unknown", "Unknown"]})
        result = compare_quality(before, after)
        self.assertEqual(result["before"]["missing_cells"], 2)
        self.assertEqual(result["after"]["missing_cells"], 0)

    def test_quality_comparison(self):
        result = compare_quality(pd.DataFrame({'x': [1, np.nan, 1]}), pd.DataFrame({'x': [1, 1]}))
        self.assertAlmostEqual(result['row_retention_pct'], 66.67, places=1)

if __name__ == '__main__':
    unittest.main()
