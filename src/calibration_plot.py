'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function
def calibration_plot(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_pred, prob_true, marker='o', label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def run_calibration_plots():
    df = pd.read_csv('data/test_dt.csv')
    y_true = df['y']
    pred_lr = df['pred_lr']
    pred_dt = df['pred_dt']

    print("Calibration plot for Logistic Regression:")
    calibration_plot(y_true, pred_lr, n_bins=5)

    print("Calibration plot for Decision Tree:")
    calibration_plot(y_true, pred_dt, n_bins=5)

    # Extra Credit
    top_50_lr = df.sort_values('pred_lr', ascending=False).head(50)
    top_50_dt = df.sort_values('pred_dt', ascending=False).head(50)
    print(f"PPV (Top 50) Logistic Regression: {round(top_50_lr['y'].mean(), 3)}")
    print(f"PPV (Top 50) Decision Tree: {round(top_50_dt['y'].mean(), 3)}")
    print(f"AUC Logistic Regression: {round(roc_auc_score(y_true, pred_lr), 3)}")
    print(f"AUC Decision Tree: {round(roc_auc_score(y_true, pred_dt), 3)}")

if __name__ == "__main__":
    run_calibration_plots()