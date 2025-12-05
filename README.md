# Bank-Account-Fraud-Detection
Here are some notebooks that I've created for the Bank Account Fraud Dataset Suite (NeurIPS 2022), which can be found here: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022. There are multiple dataset "variants" in the suite. We focus on the base variant: Base.csv.

Base.csv description (from the website): "Base dataset of the BAF suite. Synthetic account opening fraud dataset with 1M instances based on a real-world dataset. It has a "month" column to allow for temporal validation, and three protected attributes (age group, employment status, and % income) to allow for fair ML evaluation."

## High-level overview of dataset:
- Target: Fraud label (1 if fraud, 0 if legit)
- Each row represents information about a single bank account application
- Fraud incidence rate: ~1%

## Performance objective:
The fraud incidence rate in this dataset is ~1%. For rare-event detection, accuracy becomes a far less meaningful metric than precision and recall. For bank account fraud in particular, missing a fraud instance is far more financially costly than falsely flagging a non-fraud instance. Therefore, we weigh recall significantly more than precision. Given this, we chose the F2 metric as the optimization goal for our models, reflecting real-world business objectives.

## Performance results:
### XGBoost.ipynb
Our XGBoost method achieved the following results on the test data:
```
Final model F2 Score: 0.34628858170800986
Final model Recall: 0.5083391243919388
Final model Precision: 0.15220557636287974
Final model Accuracy: 0.9533488446961382
```
#### Comparison to public baselines:
We compare our XGBoost model to the baseline models provided in the most popular publicly available notebook for this dataset: https://www.kaggle.com/code/lennart4711/baselinemodels-roc. In their code, they used industry-standard models with balanced class weights to modify their objective functions to account for heavy class imbalance (~1% fraud rate into account), with zero hyperparameter searching. Note that their models are directly comparable to ours since they used an identical dataset (Base.csv) and an identical train/test split (first 6 months/last 2 months).

After modifying their code to compute F2 scores for each model, I found that they achieved:
- Logistic Regression: 0.3102
- XGBoost: 0.2925
- Random Forest: 0.2475
- Neural Network: 0.3204

Observe that our XGBoost model achieves meaningful relative performance gains of (0.34628858170800986/0.3204) - 1 = ~8% over their best baseline model and (0.34628858170800986/0.2925) - 1 = ~18% over their XGBoost model. Given that these baselines already use strong, well-established models which account for the heavy class imbalance (~1% fraud rate), and are therefore representative of realistic industry prototypes, this comparison reveals that our modeling choices offer meaningful, nontrivial performance improvements on this task.
