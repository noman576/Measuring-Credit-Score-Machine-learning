# Credit Scoring and Loan Risk Classification Using Machine Learning

A comparative machine learning analysis of lending outcomes and credit-risk indicators using peer-to-peer lending datasets from **Lending Club**, **Prosper**, and **Zopa**, with a separate **FICO-related classification** notebook.

> **Project status:** Historical exploratory research / portfolio case study. The original notebooks and saved results are retained. This repository is **not** a validated loan-origination credit scoring system and should not be used for real lending decisions.

## Objective

Explore whether borrower, loan, and repayment attributes can classify loans into good/bad outcome groups, compare five classification methods, and examine how patterns differ among datasets. A separate Lending Club experiment binarizes a *last observed* FICO range; it does not reconstruct an official FICO score.

The loan-outcome labels are defined within each notebook from observed repayment/status categories, with **0 = bad** and **1 = good**. These are project-defined labels, **not** official credit scores. Some loans are excluded based on their status. Exact definitions and exclusions differ by dataset.

## Notebooks

| Notebook | Role |
| --- | --- |
| [Credit Score of Lending Club..ipynb](Credit%20Score%20of%20Lending%20Club..ipynb) | Lending Club loan-status classification and model comparison |
| [Credit Scoring of Prosper Data..ipynb](Credit%20Scoring%20of%20Prosper%20Data..ipynb) | Prosper loan-status classification |
| [Credit Scoring of Zopa Data ..ipynb](Credit%20Scoring%20of%20Zopa%20Data%20..ipynb) | Zopa status classification |
| [FICO Score of Lending Club....ipynb](FICO%20Score%20of%20Lending%20Club....ipynb) | Binary classification using the *last observed* FICO range (threshold: 600) |

The original analyses compare **Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and Support Vector Machine** (the FICO notebook contains four of these, without an SVM model).

## Data

The full datasets are **not committed** to this repository. The original notebooks expect the following files to be placed in the repository's working directory:

| Local filename | Used by |
| --- | --- |
| `appl__accepted_20072019Q3.csv` | Lending Club and FICO |
| `LCDataDictionary.csv` | Lending Club and FICO |
| `prosperLoanData.csv` | Prosper |
| `Prosper Loan Data - Variable Definitions.csv` | Prosper |
| `data_for_loanbook_extract_2020-04-01.csv` | Zopa |

The Lending Club source referenced in the original research was a Kaggle-hosted accepted-loans dataset covering 2007–2019 Q3 (reported in the original project as approximately 2.65 million observations and 150 columns). Dataset availability, versions, licenses, and access conditions may change. Search for the exact filenames through the respective dataset provider and verify the version before running. The Prosper and Zopa notebooks also use historical exports; data files are not redistributed here. No source URLs are asserted as authoritative for a specific version without independent verification.

## Running the existing notebooks

1. Clone or download this repository.
2. Create a Python environment. For example:

   ```bash
   python -m venv .venv
   ```

3. Activate it: on Windows PowerShell, `./.venv/Scripts/Activate.ps1`; on macOS/Linux, `source .venv/bin/activate`.
4. Install the libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Obtain the five original data/dictionary files legally and place them beside the notebooks, preserving the filenames above.
6. Open with `jupyter notebook` or another Jupyter-compatible editor.

**Compatibility note:** These are original historical notebooks, not freshly rerun or verified on current package versions. Some cells use older APIs (for example, `RandomUnderSampler.fit_sample` and positional `DataFrame.drop` arguments), which may require adapting to `fit_resample` and `drop(columns=...)`. The dependency list documents library families rather than promising a reproducible pinned environment. Large data processing and model training may need significant memory and time.

## Original approach

- Explore distributions, missing data, and correlations.
- Construct dataset-specific binary target labels.
- Select and encode variables, scale numerical features, and inspect feature importance.
- Use a train/test split and class rebalancing through random undersampling.
- Fit and compare classification models.
- Report classification metrics, confusion matrices, cross-validation accuracy, and ROC-related plots.

## Results and interpretation

The original experiments evaluated multiple classification algorithms across Lending Club, Prosper, and Zopa datasets, alongside a separate FICO-related classification task. Reported model accuracy varied by dataset and algorithm. The following tables summarize historical results retained in the original notebooks; these results have not been independently reproduced or validated and should not be interpreted as prospective credit-risk prediction performance.


### Lending Club — Model Comparison

The Lending Club analysis evaluated five classification algorithms for distinguishing project-defined good and bad loan outcomes. The following metrics are taken from the original notebook's saved test-set outputs.

| Model | Test Accuracy | Reported ROC-AUC |
| --- | ---: | ---: |
| Decision Tree | 88% | 0.8834 |
| Logistic Regression | 94% | 0.9634 |
| Gaussian Naive Bayes | 93% | 0.9309 |
| **Random Forest** | **95%** | **0.9692** |
| Support Vector Machine | 93% | 0.9023 |

**Key observation:** Random Forest produced the highest reported accuracy and ROC-AUC among the evaluated models. These results are historical exploratory findings, not independently reproduced or evidence of prospective credit-risk prediction performance.

**Evaluation caveat:** Outcome-related predictors and preprocessing choices may inflate performance. The reported ROC-AUC calculations use predicted class labels rather than probability scores or continuous decision scores; therefore, they should not be interpreted as conventional score-based ROC-AUC estimates.
  

### Prosper — Model Comparison

The Prosper analysis compared five classification algorithms using historical loan data. The following values are taken from the notebook's saved outputs.

| Model | Test Accuracy | Reported ROC-AUC |
| --- | ---: | ---: |
| Decision Tree | 81% | 0.8175 |
| Logistic Regression | 90% | 0.9311 |
| **Gaussian Naive Bayes** | **91%** | 0.9121 |
| Random Forest | 89% | **0.9412** |
| Support Vector Machine | 90% | 0.8454 |

**Key observation:** Gaussian Naive Bayes achieved the highest reported test accuracy, while Random Forest had the highest saved ROC-AUC value.

**Evaluation caveat:** These are historical exploratory outputs, not independently reproduced results. The ROC-AUC values were calculated using predicted class labels, and some saved AUC values do not reconcile with the corresponding confusion matrices. Outcome-related features and preprocessing choices also limit their interpretation as prospective credit-risk performance.

### Zopa — Model Comparison

The Zopa analysis evaluated five classification algorithms using historical peer-to-peer lending data. The following results are based on the original notebook's saved test-set outputs.

| Model | Test Accuracy | Reported ROC-AUC |
| --- | ---: | ---: |
| Decision Tree | 73% | 0.7287 |
| Logistic Regression | 75% | 0.7989 |
| **Gaussian Naive Bayes** | **79%** | 0.7843 |
| Random Forest | 77% | **0.8654** |
| Support Vector Machine | 76% | 0.7537 |

**Key observation:** Gaussian Naive Bayes achieved the highest test accuracy, while Random Forest achieved the highest reported ROC-AUC. The difference highlights why classification accuracy and discrimination metrics should be considered separately.

**Evaluation caveat:** These are historical exploratory results. Random Forest accuracy is calculated from the saved confusion matrix. ROC-AUC calculations are not fully consistent across models, and class imbalance, feature selection, and preprocessing choices limit direct comparisons and prospective interpretation.


### Lending Club — FICO-Related Classification

This experiment evaluated four machine learning algorithms for classifying a binary target derived from the last observed FICO score range, using a threshold of 600. It does not predict an official or continuous FICO credit score.

| Model | Test Accuracy | Reported ROC-AUC |
| --- | ---: | ---: |
| Decision Tree | 76% | 0.7585 |
| Logistic Regression | 91% | 0.8976 |
| **Gaussian Naive Bayes** | **92%** | 0.8609 |
| Random Forest | 89% | **0.9048** |

**Key observation:** Gaussian Naive Bayes achieved the highest reported test accuracy, while Random Forest achieved the highest reported ROC-AUC.

**Evaluation caveat:** These values are historical notebook outputs, not independently reproduced results. The target is based on the last observed FICO range rather than an origination-time credit score. The saved ROC-AUC calculations use predicted labels or scores depending on the model, and preprocessing choices limit prospective interpretation.
  

### Important methodological limitations

1. **Outcome leakage and feature availability:** The Lending Club notebook includes `total_pymnt`, `recoveries`, and `out_prncp` as predictors of loan status. Prosper includes `LP_NetPrincipalLoss`. These reflect loan performance after origination, so apparent classification performance can substantially overstate the ability to predict default *before a loan is issued*.
2. **Preprocessing before train/test split:** Numerical scaling, categorical encoding, and supervised feature selection/importance analysis occur before the split in the original workflows. For trustworthy prospective evaluation, learn every preprocessing step within training folds only.
3. **Random split and sampling:** Random train/test splits do not simulate future lending cohorts; use a time-based holdout when feasible. Ensure undersampling and all tuning happen inside training folds only.
4. **ROC/AUC calculation:** Some ROC/AUC cells use hard class predictions or scores that are subsequently rounded. For properly interpreted ranking performance, compute ROC-AUC and ROC curves from continuous predicted probabilities or decision scores.
5. **Target definitions and censoring:** The dataset-specific status mappings mix completed loans and delinquent statuses; excluding `Current` loans may introduce selection effects. Carefully define an observation window and outcome horizon for a forward-looking analysis.
6. **FICO interpretation:** The FICO notebook uses the midpoint of the *last observed* FICO range, dichotomized at 600. It does not estimate a continuous FICO score or reproduce FICO's proprietary scoring algorithm.
7. **No established superiority over industry benchmarks:** The project has not demonstrated an apples-to-apples out-of-time comparison with an official FICO score or production underwriting system. Any such superiority claim would be unsupported.

The analysis is valuable as an exploration of tabular classification, feature engineering, model comparison, and differences across lending datasets; **do not use these historical accuracy figures as evidence of a deployable, fair, or compliant lending decision model**.

## Improvements for a future predictive version

To extend this research into a genuine origination-time default prediction benchmark: restrict all features to information known at application time; define a stable default horizon; split by issue date; fit preprocessing and class balancing within training folds; report bad-loan recall/precision, PR-AUC, ROC-AUC, calibration and subgroup performance; and compare with defensible, clearly defined baselines.

## Tools

Python, Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, statsmodels, and SciPy / python-dateutil as required by the original environment.

## Research use

This repository is shared for learning and portfolio demonstration. Data-provider terms and any applicable privacy, fair-lending, or financial regulations must be respected. Do not use this exploratory model for real credit eligibility or financial decisions.

## Acknowledgments

This project was developed through independent analysis, practical implementation, and the application of machine learning techniques learned from academic literature, technical documentation, and online educational resources.

The research applies data preprocessing, exploratory data analysis, feature engineering, and classification algorithms to historical lending datasets from Lending Club, Prosper, and Zopa.

I acknowledge the contributions of the dataset providers, researchers, and educational resources that supported the development of this project. Specific third-party code and resources should be credited where applicable.
  
