# encoding: utf-8
"""
Created on January 8, 2016
@author: thom.hopmans
"""
import os
import logging
from random import randint
import pandas as pd
import numpy as np
from tqdm import *

logging.basicConfig(level=logging.DEBUG)


class DataProcessor:
    """Create a recommender for blogs using collaborative filtering"""

    def __init__(self):
        self.df_articles = None
        self.df_users = None
        self.behaviour_df = None
        self.behaviour_matrix = None
        self.n_unique_users = 0

    def generate_reading_behaviour_matrix(self):
        self.load_df_articles()
        self.load_df_users()
        self.left_join_articles_on_users()
        self.filter_non_articles_from_user_observations()
        self.slice_for_relevant_columns()
        self.print_summary_of_reading_statistics()
        self.create_binary_matrix()
        return self.behaviour_df, self.behaviour_matrix, self.df_articles, self.df_users

    def load_df_articles(self):
        """Load TMT articles data"""
        self.df_articles = pd.read_csv('data\\raw\\articles\\articles.csv',
                                       encoding='utf-8',
                                       index_col=[0])
        self.df_articles['article_id'] = range(0, len(self.df_articles))
        logging.info('Number of articles: {}'.format(len(self.df_articles)))

    def load_df_users(self):
        """Load reading behaviour per user data"""
        path = 'data\\raw\\users\\'
        files = os.listdir(path)
        dtype_dict = {
            'Client id': str,
            'Pagina': str,
            'Weergaven van productdetails': int,
            'Aantal toegevoegd aan winkelwagentje': int,
            'Unieke aankopen': int
        }
        dataframes = [pd.read_csv(path + fname, encoding='utf-8', sep=',', dtype=dtype_dict) for fname in files]
        self.df_users = pd.concat(dataframes, axis=0)
        self.df_users.columns = ['client_id', 'link', 'pageviews', 'q_pageviews', 'reads']
        print(self.df_users['client_id'].head())
        self.df_users = self.df_users.groupby(['client_id', 'link']).sum().reset_index()
        self.df_users['is_read'] = self.df_users.apply(lambda x: self.get_is_read_value(x), axis=1)
        logging.info('Number of user observations: {}'.format(len(self.df_users)))

    @staticmethod
    def get_is_read_value(x):
        if x['reads'] > 0:
            val = 1
        elif x['q_pageviews'] > 0:
            val = 0.50
        elif x['pageviews'] > 0:
            val = 0.1
        else:
            val = 0
        return val

    def left_join_articles_on_users(self):
        """Add the article id to each user observation. We will use this to clean the data by removing non-article
        observations and changed links that therefore cannot be matched to an article."""
        self.df_users = self.df_users.merge(self.df_articles[['link', 'article_id']], on='link', how='left')

    def filter_non_articles_from_user_observations(self):
        """Filter out non-articles, i.e. author page"""
        condition = self.df_users['article_id'].isnull()
        self.df_users = self.df_users[~condition]
        logging.info('Removed {} user observations corresponding to non-articles. '
                     '{} user observations remaining.'.format(condition.sum(), len(self.df_users)))
        # TODO: Filter out few strange observations, where article is read but there is no pageview

    def slice_for_relevant_columns(self):
        relevant_columns = ['client_id', 'article_id', 'is_read']
        self.df_users = self.df_users[relevant_columns]

    def print_summary_of_reading_statistics(self):
        self.n_unique_users = len(self.df_users['client_id'].unique())
        logging.info('Number of unique users: {}'.format(self.n_unique_users))

        df_users_client_id_count = self.df_users.groupby(['client_id']).count()
        df_users_client_id_count = df_users_client_id_count[df_users_client_id_count['article_id'] > 1]
        perc_users_multiple_articles = self.perc_of_unique_users(len(df_users_client_id_count))
        logging.info('Percentage of unique users that opened multiple articles: {}%'.format(perc_users_multiple_articles))

        df_users_client_id_sum = self.df_users.groupby(['client_id']).sum()
        df_users_with_at_least_one_read = df_users_client_id_sum[df_users_client_id_sum['is_read'] >= 1]
        df_users_with_at_least_two_reads = df_users_client_id_sum[df_users_client_id_sum['is_read'] >= 2]
        logging.info('Number of unique users with at least one full read: {}'.format(len(df_users_with_at_least_one_read)))
        logging.info('Number of unique users with at least two full reads: {}'.format(len(df_users_with_at_least_two_reads)))
        perc_users_multiple_articles_read = self.perc_of_unique_users(len(df_users_with_at_least_two_reads))
        logging.info('Percentage of unique users that read multiple articles: {}%'.format(perc_users_multiple_articles_read))

    def perc_of_unique_users(self, n):
        perc = float(n) / self.n_unique_users
        return np.round(perc * 100, 2)

    def create_binary_matrix(self):
        self.add_articles_as_columns()
        self.mark_read_articles_per_user()
        del self.df_users['article_id']
        del self.df_users['is_read']
        self.behaviour_df = self.df_users.groupby('client_id').sum()
        self.behaviour_matrix = self.behaviour_df.values

    def add_articles_as_columns(self):
        article_id_list = sorted(list(self.df_users['article_id'].unique()))
        for id in article_id_list:
            self.df_users['article_'+str(id)] = 0

    def mark_read_articles_per_user(self):
        for index, row in tqdm(self.df_users.iterrows(), total=len(self.df_users)):
            self.df_users.set_value(index, 'article_'+str(row['article_id']), row['is_read'])

    def get_recommendations_for_random_users(self):
        for i in range(100):
            user_id = randint(1, len(self.behaviour_df.index))
            self.get_recommendation_for_user(user_id)


if __name__ == "__main__":
    DataProcessor().generate_reading_behaviour_matrix()
