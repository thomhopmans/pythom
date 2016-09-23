# encoding: utf-8
"""
Created on September 22, 2016
@author: thom.hopmans
"""

import pandas as pd
import statsmodels.api as sm
from util import parse_unixtstamp_as_datetime


class BusinessCaseSolver(object):

    def __init__(self):
        self.df_sessions = None
        self.df_engagements = None
        self.df = None

    def run(self):
        # Read data
        self.df_sessions = pd.read_csv('data\\df_sessions.csv',
                                       parse_dates={'datetime': [2]})
        relevant_columns = ['datetime', 'user_id', 'session_number', 'pageviews']
        self.df_sessions = self.df_sessions[relevant_columns]

        self.df_engagements = pd.read_csv('data\\df_engagements.csv',
                                          sep=';',
                                          parse_dates={'datetime': [2]},
                                          date_parser=parse_unixtstamp_as_datetime
                                          )
        relevant_columns = ['datetime', 'user_id']
        self.df_engagements = self.df_engagements[relevant_columns]

        # Get only the first engagement of each user
        self.df_engagements.sort_values(['user_id', 'datetime'], ascending=True, inplace=True)
        self.df_engagements = self.df_engagements.groupby(['user_id'])['datetime'].first().reset_index()

        # Merge data sets on user_id
        self.df = pd.merge(self.df_sessions,
                           self.df_engagements,
                           on='user_id',
                           how='left',
                           suffixes=('_session', '_first_engagement'))

        # Delete sessions that are after the first engagement, not necessary for this analysis.
        condition = self.df['datetime_first_engagement'] >= self.df['datetime_session']
        self.df = self.df[condition].copy()
        print self.df.head(10)


        # Logistic regression
        # logit = sm.Logit(self.df_both['conversion'], self.df_both['pageviews_y'])
        # result = logit.fit()
        # print result.summary()

if __name__ == "__main__":
    BusinessCaseSolver().run()
