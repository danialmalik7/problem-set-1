'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():
    # PART 1
    etl.run_etl()

    # PART 2
    preprocessing.run_preprocessing()

    # PART 3
    logistic_regression.run_logistic_regression()

    # PART 4
    decision_tree.run_decision_tree()

    # PART 5
    calibration_plot.run_calibration_plots()

if __name__ == "__main__":
    main()
