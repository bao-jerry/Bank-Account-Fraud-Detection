# Bank-Account-Fraud-Detection
Here are some notebooks that I've created for the Bank Account Fraud Dataset Suite (NeurIPS 2022), which can be found here: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022. There are multiple dataset "variants" in the suite. We focus on the base variant: **Base.csv**.
### Legend:
- XGBoost.ipynb: Analyzing the dataset with XGBoost modelling.
- SVMs.ipynb: Analyzing the dataset with linear-kernel SVMs and RBF-kernel SVM ensembles.
### Usage guide:
1. Download **Base.csv** from https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 and place it in the same directory as the notebooks.
2. Run the cells in the notebooks.

## High-level overview of **Base.csv**:
- Target: Fraud label (1 if fraud, 0 if legit)
- Each row represents information about a single bank account application.
- Fraud incidence rate: ~1%
- Description (from the website): "Base dataset of the BAF suite. Synthetic account opening fraud dataset with 1M instances based on a real-world dataset. It has a "month" column to allow for temporal validation, and three protected attributes (age group, employment status, and % income) to allow for fair ML evaluation."

## Performance objective:
The fraud incidence rate in this dataset is ~1%. For rare-event detection, accuracy becomes a far less meaningful metric than precision and recall. For bank account fraud in particular, missing a fraud instance is far more financially costly than falsely flagging a non-fraud instance. Therefore, we weigh recall significantly more than precision. Given this, we chose the F2 score as the optimization goal for our models, reflecting real-world business objectives.

## Performance results (from XGBoost.ipynb):
Our XGBoost method achieved the following results on the test data:
```
Final model F2 Score: 0.34628858170800986
Final model Recall: 0.5083391243919388
Final model Precision: 0.15220557636287974
Final model Accuracy: 0.9533488446961382
```
#### Comparison to public baselines:
We compare our XGBoost model to the baseline models provided in the most popular publicly available notebook for this dataset: https://www.kaggle.com/code/lennart4711/baselinemodels-roc. In their code, they used industry-standard models with balanced class weights to modify their objective functions to account for heavy class imbalance (~1% fraud rate into account), with zero hyperparameter searching. Note that their models are directly comparable to ours since they used an identical dataset (**Base.csv**) and an identical train/test split (first 6 months' data/last 2 months' data).

After modifying their code to compute F2 scores for each model, I found that they achieved:
- Logistic Regression: 0.3102
- XGBoost: 0.2925
- Random Forest: 0.2475
- Neural Network: 0.3204

Observe that our XGBoost model achieves meaningful relative performance gains of (0.34628858170800986/0.3204) - 1 = **~8%** over their best baseline model and (0.34628858170800986/0.2925) - 1 = **~18%** over their XGBoost model. Given that these baselines already use strong, well-established models which account for the heavy class imbalance (~1% fraud rate), and are therefore representative of realistic industry prototypes, this comparison reveals that our modeling choices offer meaningful, nontrivial performance improvements on this task.

## Interpretable feature insights (from SVMs.ipynb):
From our linear-kernel SVM, we filtered for the standardized features with the strongest coefficient signals (absolute value >= 0.1). Here are the key takeaways:

### Positive signals for fraud:
- According to this SVM model, if information about the "number of months in previous registered address of the applicant" is missing, this is a relatively strong signal for fraud. One potential reason is that fraudsters tend to avoid providing a traceable address history.
- According to this SVM model, the device OS being windows is a relatively strong signal for fraud. One possible explanation is that fraudsters often operate from cheap and widely available Windows environments.

### Negative signals for fraud:
- According to this SVM, the user keeping the session alive on session logout is a negative indicator of fraud. A potential reason for this is that fraudsters tend to avoid being connected to the system longer than they have to be.
- According to this SVM model, the similarity in the applicant's name to the email name is a negative indicator for fraud. One possible explanation is that fraudsters often use randomly generated or non-identifying email handles.
- According to this SVM model, the validity of the home phone number is a negative indicator of fraud. One possible explanation is that fraudsters often supply fake or burner phone numbers.
- According to this SVM model, the applicant having other cards with the same banking company is a negative signal of fraud. This may be because having multiple cards with the same company under the same identity increases the fraudster's detectability, so fraudsters generally avoid this.
- Housing status being BE, BB, and BC are negative indicators of fraud according to this model. However, given that BE, BB, and BC are anonymized values, we can't say much more beyond this.

