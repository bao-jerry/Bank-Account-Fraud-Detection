# Bank-Account-Fraud-Detection
Here are some notebooks that I've created for the Bank Account Fraud Dataset Suite (NeurIPS 2022), which can be found here: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022. There are multiple dataset "variants" in the suite. We focus on the base variant: **Base.csv**. All code was tested with Python version 3.12.12.
### High-level overview of **Base.csv**:
- **Target:** Fraud label (1 if fraud, 0 if legit)
- **Features:** Each row represents information about a single bank account application.
- **Fraud incidence rate:** ~1%
- **Description (from the website):** "Base dataset of the BAF suite. Synthetic account opening fraud dataset with 1M instances based on a real-world dataset. It has a "month" column to allow for temporal validation, and three protected attributes (age group, employment status, and % income) to allow for fair ML evaluation."
### Legend:
- **Exploratory_Data_Analysis.ipynb:** Feature correlation analysis, linear SVM feature coefficient analysis, feature heat map analysis. With a focus on interpretable insights.
- **Performance_Modeling.ipynb:** XGBoost modelling of the performance objective (see "Performance objective" below), XGBoost global SHAP feature analysis
### Usage guide:
1. Download **Base.csv** from https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 and place it in the same directory as the notebooks.
2. Run the cells in the notebooks.

## Performance objective:
Let:
- **TPR** = true positive rate (recall)  
- **FPR** = false positive rate  
- Positive rate **π = P(Y = 1)** (rate of fraud)  
- Negative rate **1 − π = P(Y = 0)** (rate of non-fraud)
- **C_R** = expected *relative* costs associated with manually reviewing a case flagged by the model
- **C_FN** = expected *relative* cost of a false negative (failing to detect fraud)

Then the population-normalized expected loss is:

E[Loss] = C_R · (TPR · π + FPR · (1 − π)) + C_FN · (1 − TPR) · π

Key modeling assumptions:
- After the model detects fraud, the case undergoes manual review.
- Manual review is treated as the ground-truth adjudication process (assumed to make the final decision for whether fraud occurred).
- Costs associated with manual review are approximately equal for false positives and true positives.
- Since this is a synthetic dataset, we do not have access to true values for C_R and C_FN. Thus, for the purposes of this hypothetical modeling scenario, we choose illustrative values for C_R and C_FN. We estimate the average cost of manual review plus associated operational and customer-friction costs to be approximately $40–$80 per reviewed account. We estimate the expected fraud loss per missed fraud case to be approximately $2,000–$5,000. This implies a plausible range for C_FN/C_R of roughly 25 to 125. For this investigation, we will adopt a balanced scenario using the midpoint of this range, with C_FN/C_R = 75. Therefore, we use C_R = 1 and C_FN = 75.

The final performance objective, which we will aim to minimize, is then:

<ins>**E[Loss] = (TPR · π + FPR · (1 − π)) + 75(1 − TPR) · π**</ins>

To simulate real-world effects of temporal instability/concept drift, our test dataset will consist of the final 2 months of data.

## Performance results (from Performance_Modeling.ipynb):
We compare our final XGBoost model to standard baseline models provided in the most popular published notebook for this dataset: https://www.kaggle.com/code/lennart4711/baselinemodels-roc. 

The baseline models include logistic regression, XGBoost, random forest, and neural network models. Each baseline model was implemented with adapted class weights to account for heavy class imbalance (~1% fraud rate) and did not undergo additional exhaustive hyperparameter search. Note that these baseline models are directly comparable to ours since they use an identical dataset (**Base.csv**) and an identical train/test split (first 6 months' data/last 2 months' data).

**Our final XGBoost model achieves the following performance gains:**
- ***~30% reduction in expected financial losses compared to the most performant baseline model.***
- ***~30-50% reduction in expected financial losses compared to all baseline models.***
- ***~65% reduction in expected financial losses compared to the "do nothing" strategy.***

Our model's large performance gains indicate that our modeling decisions led to significant, nontrivial improvements for this task.

## Performance Robustness (from Performance_Modeling.ipynb):
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/65dfe835-5e6a-4750-9ed1-113088bb6df6" />

Although we fixed C_FN = 75 for our official objective function, in practice, the estimate for C_FN may be inexact. In this section, we verify how robust our model is to perturbations in C_FN.

In the above graph, we plot each model's E[Loss] in the C_FN range [25, 125].

Observations:
- When C_FN is very low (25), our final XGBoost model is comparable with other baseline models, yet still competitive.
- When C_FN is moderately low, medium, or high-valued, our final XGBoost model strictly outperforms all baseline models by a significant margin.
- In all cases, all models strictly outperform the "do nothing" strategy by a significant margin.

We conclude that our model is robust to deviations in C_FN.

## Exploratory Data Analysis:
- For **feature correlation analysis, linear SVM feature coefficient analysis, and feature heat map analysis**, see Exploratory_Data_Analysis.ipynb.
- For **XGBoost Global SHAP feature analysis**, see Performance_Modeling.ipynb.
