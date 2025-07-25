'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Save the dataframe for PART 4 and PART 5 in `data/`
'''

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

def run_logistic_regression():
    df_arrests = pd.read_csv('data/df_arrests.csv')
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    X = df_arrests[features]
    y = df_arrests['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
    param_grid = {'C': [0.01, 0.1, 1]}
    lr_model = LogisticRegression(max_iter=500)
    gs_cv = GridSearchCV(lr_model, param_grid, cv=5)
    gs_cv.fit(X_train, y_train)

    best_c = gs_cv.best_params_['C']
    print(f"Optimal value for C: {best_c}")
    if best_c == 0.01:
        print("This means the most regularization.")
    elif best_c == 1:
        print("This means the least regularization.")
    else:
        print("This is in the middle.")

    X_test['y'] = y_test.values
    X_test['pred_lr'] = gs_cv.predict_proba(X_test[features])[:, 1]
    X_test.to_csv('data/test_lr.csv', index=False)
    print("Part 3 Test set with predictions saved to data/test_lr.csv")

if __name__ == "__main__":
    run_logistic_regression()
