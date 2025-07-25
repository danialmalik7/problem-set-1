'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Your code here
def run_decision_tree():
    df_test = pd.read_csv('data/test_lr.csv')
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    X_test = df_test[features]
    y_test = df_test['y']

    param_grid_dt = {'max_depth': [2, 4, 6]}
    dt_model = DecisionTreeClassifier()
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    gs_cv_dt.fit(X_test, y_test)

    best_depth = gs_cv_dt.best_params_['max_depth']
    print(f"Optimal value for max_depth: {best_depth}")
    if best_depth == 2:
        print("This means the most regularization (simplest tree).")
    elif best_depth == 6:
        print("This means the least regularization (most complex tree).")
    else:
        print("This is in the middle.")

    df_test['pred_dt'] = gs_cv_dt.predict_proba(X_test)[:, 1]
    df_test.to_csv('data/test_dt.csv', index=False)
    print("Part 4 - Predictions saved to data/test_dt.csv")

if __name__ == "__main__":
    run_decision_tree()