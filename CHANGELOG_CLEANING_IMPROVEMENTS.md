# Adaptive Cleaning Improvements

## Version 1.2

- Tightened identifier detection so high-cardinality continuous measurements are not automatically treated as IDs.
- Added per-numeric-column masking benchmarks comparing mean, median, and KNN imputation.
- Method selection now uses measured MAE, with RMSE as a tie-breaker; if a benchmark cannot run, a documented simple fallback is used.
- Added separate completeness and duplicate-free quality components while preserving the existing indicative score formula.
- Added anomaly percentage, detection-method metadata, and a preview of flagged records.
- Added a Flask download route for the complete set of anomaly-flagged records.
- Expanded the HTML cleaning report with component scores, imputation benchmark metrics, synthetic benchmark comparisons, and anomaly review details.
- Expanded the PDF cleaning section with the score components, per-column benchmark results, synthetic baseline comparison, and anomaly count/percentage.
- Flagged rows continue to be retained; the system does not automatically delete anomalies.

## Validation

- Unit tests: 7 tests pass.
- Titanic smoke test: 891 rows retained, missing cells reduced from 866 to 0, 27 rows flagged, and the indicative score changed from 95.55% to 100.00%.
- Jinja cleaning-report template parses successfully.
- Full Flask route/PDF integration could not be executed in the build environment because Flask is not installed there. Run the app in the project's normal virtual environment to validate the browser download routes and generated PDF.


## V4 waiting-list implementation (1.3)

- Initial profiling now counts common missing placeholders (`?`, `N/A`, `NULL`, etc.) as missing before cleaning, so the before-quality score reflects them.
- Numeric imputation now compares Mean, Median, KNN, MissForest-style Random-Forest iterative imputation, and MICE-style Bayesian iterative imputation with repeated masked-value MAE/RMSE validation.
- Anomaly detection now compares Isolation Forest, Local Outlier Factor, DBSCAN, and One-Class SVM, then applies the detector with the strongest weighted consensus agreement.
- Flagged records now carry `_auto_clean_anomaly_reason`, with the largest robust within-group feature deviations shown as review clues.
- The cleaning report and PDF include comparison metrics, selected algorithms, selection caveats, and example anomaly explanations. Audit metadata is excluded from EDA and predictive ML features.
- Anomaly selection is unsupervised consensus only; it is not a claim of true accuracy. Labelled anomaly examples are needed to estimate precision and recall.
