# encoding: utf-8
"""
Created on January 8, 2016
@author: thom.hopmans
"""
import logging
from random import randint
import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd

logging.basicConfig(level=logging.DEBUG)


class ModelFitter:
    """Create a recommender for blogs using collaborative filtering"""

    def __init__(self, behaviour_df, behaviour_matrix, df_articles, df_users):
        self.df_articles = df_articles
        self.df_users = df_users
        self.behaviour_df = behaviour_df
        self.behaviour_matrix = behaviour_matrix
        self.n_unique_users = 0
        self.X_hat = None

    def fit_model(self):
        self.apply_uv_decomposition()
        self.get_recommendations_for_random_users()

    def apply_uv_decomposition(self):
        U, Sigma, VT = randomized_svd(self.behaviour_matrix,
                                      n_components=15,
                                      n_iter=10,
                                      random_state=None)
        print(U.shape)
        print(VT.shape)
        self.X_hat = np.dot(U, VT)  # U * np.diag(Sigma)

    def get_recommendations_for_random_users(self):
        for i in range(100):
            user_id = randint(1, len(self.behaviour_df.index))
            self.get_recommendation_for_user(user_id)

    def get_recommendation_for_user(self, row_index):
        print("---------------")
        self.get_titles_user_has_read(row_index)
        print("Recommendations:")
        recommendation_vector = self.X_hat[row_index, :]
        highest_ids = np.argpartition(recommendation_vector, -5)[-5:]
        for article_id in highest_ids:
            print(' > ', article_id, self.get_title_by_article_id(article_id))

    def get_titles_user_has_read(self, row_index):
        print("User {} has read:".format(row_index))
        user_series = self.behaviour_df.iloc[row_index]
        condition = user_series > 0
        read = user_series.ix[condition]
        print(read)
        if read.empty is False:
            for article_id, value in read.iteritems():
                article_id = int(float((article_id.split("_")[1])))
                print(' > ', article_id, self.get_title_by_article_id(article_id), '(value:', value, ')')

    def get_title_by_article_id(self, id):
        return self.df_articles.ix[self.df_articles['article_id'] == id]['title'].values[0]

if __name__ == "__main__":
    CollaborativeFiltering().run()
