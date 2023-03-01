"""
This code solves the fiction business problem as described in the blog on The Marketing Technologist. This is done
by loading the data, cleaning the data, applying conversion logic, running a logistic regression on the cleaned
data and finally, visualizing the results.
"""
from datetime import datetime

import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def run():
    """ Run the full script step-by-step"""
    # Load data
    df_sessions = read_sessions_data()
    df_engagements = read_engagements_data()
    print(df_sessions.head())
    print(df_engagements.head())
    # Transform data
    df_engagements = filter_for_first_engagements(df_engagements)
    df = merge_dataframes_on_user_id(df_sessions, df_engagements)
    df = remove_sessions_after_first_engagement(df)
    df = add_conversion_metric(df)
    df = add_pageviews_cumsum(df)
    df.to_csv('output\\df_transformed.csv')
    # Fit model using logistic regression
    logistic_regression_results = run_logistic_regression(df)
    predict_probabilities(logistic_regression_results)
    # Visualize results
    visualize_results(df)


def read_sessions_data():
    # Read sessions data
    df_sessions = pd.read_csv(filepath_or_buffer='data\\df_sessions.csv',
                              parse_dates={'datetime': [2]})
    relevant_columns = ['datetime', 'user_id', 'session_number', 'pageviews']
    df_sessions = df_sessions[relevant_columns]
    return df_sessions


def read_engagements_data():
    # Read engagements data
    df_engagements = pd.read_csv(filepath_or_buffer='data\\df_engagements.csv',
                                 sep=';',
                                 parse_dates={'datetime': [2]},
                                 date_parser=parse_unixtstamp_as_datetime)
    relevant_columns = ['datetime', 'user_id']
    df_engagements = df_engagements[relevant_columns]
    return df_engagements


def parse_unixtstamp_as_datetime(unix_tstamp):
    datetime_obj = datetime.fromtimestamp(int(unix_tstamp))
    return datetime_obj


def filter_for_first_engagements(df_engagements):
    # Get only the first engagement of each user
    df_engagements.sort_values(['user_id', 'datetime'], ascending=True, inplace=True)
    df_engagements = df_engagements.groupby(['user_id'])['datetime'].first().reset_index()
    return df_engagements


def merge_dataframes_on_user_id(df_sessions, df_engagements):
    # Merge data sets on user_id
    df = pd.merge(df_sessions,
                  df_engagements,
                  on='user_id',
                  how='left',
                  suffixes=('_session', '_first_engagement'))
    return df


def remove_sessions_after_first_engagement(df):
    # Delete sessions that are after the first engagement, not necessary for this analysis.
    condition = df['datetime_first_engagement'] >= df['datetime_session']
    df = df[condition].copy()
    return df


def add_conversion_metric(df):
    # Add conversion metric
    df['is_conversion'] = False
    # Get row indices of sessions with engagements and set is_conversion to true
    indices = df.groupby(['user_id']).apply(lambda x: x['datetime_session'].idxmax())
    df.ix[indices, 'is_conversion'] = True
    return df


def add_pageviews_cumsum(df):
    # Add cumulative sum of pageviews
    df['pageviews_cumsum'] = df.groupby('user_id')['pageviews'].cumsum()
    return df


def run_logistic_regression(df):
    # Logistic regression
    X = df['pageviews_cumsum']
    X = sm.add_constant(X)
    y = df['is_conversion']
    logit = sm.Logit(y, X)
    logistic_regression_results = logit.fit()
    print(logistic_regression_results.summary())
    return logistic_regression_results


def predict_probabilities(logistic_regression_results):
    # Predict the conversion probability for 0 up till 50 pageviews
    X = sm.add_constant(range(0, 50))
    y_hat = logistic_regression_results.predict(X)
    df_hat = pd.DataFrame(zip(X[:, 1], y_hat))
    df_hat.columns = ['X', 'y_hat']
    p_conversion_25_pageviews = df_hat.ix[25]['y_hat']
    print("")
    print(f"The probability of converting after 25 pageviews is {p_conversion_25_pageviews}")


def visualize_results(df):
    # Visualize logistic curve using seaborn
    sns.set(style="darkgrid")
    sns.regplot(x="pageviews_cumsum",
                y="is_conversion",
                data=df,
                logistic=True,
                n_boot=500,
                y_jitter=.01,
                scatter_kws={"s": 60})
    sns.set(font_scale=1.3)
    sns.plt.title('Logistic Regression Curve')
    sns.plt.ylabel('Conversion probability')
    sns.plt.xlabel('Cumulative sum of pageviews')
    sns.plt.subplots_adjust(right=0.93, top=0.90, left=0.10, bottom=0.10)
    sns.plt.show()


if __name__ == "__main__":
    # Run the final program
    run()
