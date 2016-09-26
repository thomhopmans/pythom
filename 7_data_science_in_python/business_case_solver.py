# encoding: utf-8
"""
Created on September 22, 2016
@author: thom.hopmans
"""
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from common.base import parse_unixtstamp_as_datetime


class BusinessCaseSolver(object):
    """
    This code solves the fiction business problem as described in the blog on The Marketing Technologist. This is done
    by loading the data, cleaning the data, applying conversion logic, running a logistic regression on the cleaned
    data and finally, visualizing the results.
    """

    def __init__(self):
        self.path_input_sessions = 'data\\df_sessions.csv'
        self.path_input_engagements = 'data\\df_engagements.csv'
        self.df_sessions = None
        self.df_engagements = None
        self.df = None
        self.logistic_regression_results = None

    def run(self):
        """ Run the full script """
        self.read_sessions_data()
        self.read_engagements_data()
        self.filter_for_first_engagements()
        self.merge_dataframes_on_user_id()
        self.remove_sessions_after_first_engagement()
        self.add_conversion_metric()
        self.add_pageviews_cumsum()
        self.run_logistic_regression()
        self.predict_probabilities()
        self.visualize_results()

    def read_sessions_data(self):
        # Read sessions data
        self.df_sessions = pd.read_csv(filepath_or_buffer=self.path_input_sessions,
                                       parse_dates={'datetime': [2]})
        relevant_columns = ['datetime', 'user_id', 'session_number', 'pageviews']
        self.df_sessions = self.df_sessions[relevant_columns]

    def read_engagements_data(self):
        # Read engagements data
        self.df_engagements = pd.read_csv(filepath_or_buffer=self.path_input_engagements,
                                          sep=';',
                                          parse_dates={'datetime': [2]},
                                          date_parser=parse_unixtstamp_as_datetime)
        relevant_columns = ['datetime', 'user_id']
        self.df_engagements = self.df_engagements[relevant_columns]

    def filter_for_first_engagements(self):
        # Get only the first engagement of each user
        self.df_engagements.sort_values(['user_id', 'datetime'], ascending=True, inplace=True)
        self.df_engagements = self.df_engagements.groupby(['user_id'])['datetime'].first().reset_index()

    def merge_dataframes_on_user_id(self):
        # Merge data sets on user_id
        self.df = pd.merge(self.df_sessions,
                           self.df_engagements,
                           on='user_id',
                           how='left',
                           suffixes=('_session', '_first_engagement'))

    def remove_sessions_after_first_engagement(self):
        # Delete sessions that are after the first engagement, not necessary for this analysis.
        condition = self.df['datetime_first_engagement'] >= self.df['datetime_session']
        self.df = self.df[condition].copy()

    def add_conversion_metric(self):
        # Add conversion metric
        self.df['is_conversion'] = False
        # Get row indices of sessions with engagements and set is_conversion to true
        indices = self.df.groupby(['user_id']).apply(lambda x: x['datetime_session'].idxmax())
        self.df.ix[indices, 'is_conversion'] = True

    def add_pageviews_cumsum(self):
        # Add cumulative sum of pageviews
        self.df['pageviews_cumsum'] = self.df.groupby('user_id')['pageviews'].cumsum()
        print self.df.head(5)

    def run_logistic_regression(self):
        # Logistic regression
        X = self.df['pageviews_cumsum']
        X = sm.add_constant(X)
        y = self.df['is_conversion']
        logit = sm.Logit(y, X)
        self.logistic_regression_results = logit.fit()
        print self.logistic_regression_results.summary()

    def predict_probabilities(self):
        # Predict the conversion probability for 0 up till 50 pageviews
        X = sm.add_constant(range(0, 50))
        y_hat = self.logistic_regression_results.predict(X)
        df_hat = pd.DataFrame(zip(X, y_hat))
        df_hat.columns = ['X', 'y_hat']
        print df_hat

    def visualize_results(self):
        # Visualize logistic curve using seaborn
        sns.regplot(x="pageviews_cumsum",
                    y="is_conversion",
                    data=self.df,
                    logistic=True,
                    n_boot=500,
                    y_jitter=.01)
        sns.plt.show()

if __name__ == "__main__":
    BusinessCaseSolver().run()
