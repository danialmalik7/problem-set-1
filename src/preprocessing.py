'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statement to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statement to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number of arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statement to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' and df_arrests.head()
- Save `df_arrests` to data/df_arrests.csv for use in PART 3.
'''

import pandas as pd

def run_preprocessing():
    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('data/arrest_events_raw.csv')

    # Merge data on person_id
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    # Convert to datetime
    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'], errors='coerce')
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'], errors='coerce')

    # Merge for comparisons
    merged = df_arrests[['person_id', 'arrest_date_univ']].merge(
        arrest_events[['person_id', 'arrest_date_event', 'charge_degree']], on='person_id', how='left'
    )

    merged['arrest_date_event'] = pd.to_datetime(merged['arrest_date_event'], errors='coerce')
    merged['arrest_date_univ'] = pd.to_datetime(merged['arrest_date_univ'], errors='coerce')
    merged['days_diff'] = (merged['arrest_date_event'] - merged['arrest_date_univ']).dt.days

    # y = felony arrest within 365 days
    mask_future = (merged['days_diff'] >= 1) & (merged['days_diff'] <= 365) & (merged['charge_degree'].str.lower() == 'felony')
    rearrest_ids = merged.loc[mask_future, 'person_id'].unique()
    df_arrests['y'] = df_arrests['person_id'].isin(rearrest_ids).astype(int)

    print("What share of arrestees were rearrested for a felony crime in the next year?")
    print(round(df_arrests['y'].mean(), 3))

    # current_charge_felony
    df_arrests['current_charge_felony'] = (df_arrests['charge_degree'].str.lower() == 'felony').astype(int)
    print("What share of current charges are felonies?")
    print(round(df_arrests['current_charge_felony'].mean(), 3))

    # num_fel_arrests_last_year
    mask_past = (merged['days_diff'] <= -1) & (merged['days_diff'] >= -365) & (merged['charge_degree'].str.lower() == 'felony')
    past_counts = merged.loc[mask_past].groupby('person_id').size()
    df_arrests['num_fel_arrests_last_year'] = df_arrests['person_id'].map(past_counts).fillna(0).astype(int)

    print("What is the average number of felony arrests in the last year?")
    print(round(df_arrests['num_fel_arrests_last_year'].mean(), 3))

    print(df_arrests.head())

    df_arrests.to_csv('data/df_arrests.csv', index=False)
    print("PART 2 complete: Preprocessed data saved to data/df_arrests.csv")

if __name__ == "__main__":
    run_preprocessing()