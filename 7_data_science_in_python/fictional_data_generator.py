# encoding: utf-8
"""
Created on September 22, 2016
@author: thom.hopmans
"""

import logging
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from util import parse_datetime_as_unixtstamp

logging.basicConfig(level=logging.DEBUG)

N_USERS = 10
DATE_START = datetime(2016, 9, 1)


class FictionalDataGenerator:
    """
    Create a fictional session and engagements dataset for the Python Data Science crash course.

    Note: this is not the fastest way to create the desired fictional dataframes as it contains several slow functions.
    This code is designed in a manner such that new Data Scientists starting in Python should be able to understand it.
    """

    def __init__(self):
        self.df = pd.DataFrame()
        self.df_sessions = None
        self.df_engagements = None
        self.engagements_mean = 20
        self.engagements_stdev = 8

    def run(self):
        """ Run the full script """
        self.fill_dataframe_with_random_users_and_sessions()
        self.split_dataframe_in_sessions_and_engagements()
        self.save_dataframes()

    def split_dataframe_in_sessions_and_engagements(self):
        self.extract_sessions_dataframe()
        self.extract_engagements_dataframe()

    def extract_sessions_dataframe(self):
        relevant_columns = ['user_id', 'session_number', 'session_start_date', 'unix_timestamp', 'campaign_id',
                            'domain', 'entry', 'referral', 'pageviews', 'transactions']
        self.df_sessions = self.df[relevant_columns]

    def extract_engagements_dataframe(self):
        condition = self.df['has_engagement'] == True
        self.df_engagements = self.df.ix[condition]
        relevant_columns = ['user_id', 'site_id', 'engagement_unix_timestamp', 'engagement_type', 'custom_properties']
        self.df_engagements = self.df_engagements[relevant_columns]

    def save_dataframes(self):
        self.df.set_index('user_id').to_csv('data\\df.csv', sep=',')
        self.df_sessions.set_index('user_id').to_csv('data\\df_sessions.csv', sep=',')
        self.df_engagements.set_index('user_id').to_csv('data\\df_engagements.csv', sep=';')

    def fill_dataframe_with_random_users_and_sessions(self):
        for i in xrange(N_USERS):
            user_id = str(uuid.uuid4())
            self.generate_random_sessions_per_user(user_id)

    def generate_random_sessions_per_user(self, uid):
        n_sessions = int(np.ceil(np.random.rand(1) * 10))
        session_start_date = DATE_START
        sum_pageviews = 0

        for i in range(1, n_sessions+1):
            session_start_date = session_start_date + timedelta(days=int(np.ceil(np.random.rand(1) * 3)),
                                                                hours=int(np.ceil(np.random.rand(1) * 24)),
                                                                minutes=int(np.ceil(np.random.rand(1) * 60)),
                                                                seconds=int(np.ceil(np.random.rand(1) * 60)))
            n_pageviews = int(np.ceil(np.random.exponential(1) * 5))
            sum_pageviews += n_pageviews

            engagement_bool = self.has_engagement_in_session(sum_pageviews)

            session_dict = {
                'user_id': uid,
                'session_number': i,
                'session_start_date': session_start_date.strftime('%Y-%m-%d %H:%I:%S'),
                'unix_timestamp': parse_datetime_as_unixtstamp(session_start_date),
                'site_id': 596,
                'domain': 'www.themarketingtechnologist.co',
                'referral': 'www.google.nl',
                'entry': 'www.themarketingtechnologist.co',
                'campaign_id': int(np.ceil(np.random.rand(1) * 1000)),
                'transactions': 0,
                'pageviews': n_pageviews,
                'has_engagement': engagement_bool
            }

            if engagement_bool:
                # Add engagement metrics
                engagement_timestamp = session_start_date + timedelta(minutes=int(np.ceil(np.random.rand(1) * 20)))
                session_dict['engagement_unix_timestamp'] = parse_datetime_as_unixtstamp(engagement_timestamp)
                session_dict['engagement_type'] = 'newsletter_subscription'
                session_dict['custom_properties'] = str({'gender': 'male', 'age': 29, 'country': 'Netherlands'})
                # Reset sum_pageviews if an engagement took place in session
                sum_pageviews = 0

            # Append session row to total DataFrame
            df_row = pd.DataFrame(session_dict, index=[0])
            self.df = self.df.append(df_row, ignore_index=True)

    def has_engagement_in_session(self, pageviews):
        critical_value = np.random.normal(self.engagements_mean, self.engagements_stdev)
        if pageviews > critical_value:
            return True
        else:
            return False

if __name__ == "__main__":
    FictionalDataGenerator().run()
