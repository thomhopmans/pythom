import logging

from data_processor import DataProcessor
from model_fitter import ModelFitter

logging.basicConfig(level=logging.DEBUG)


class CollaborativeFiltering:
    """Create a recommender for blogs using collaborative filtering"""

    def run(self):
        """Run the Scrape TMT articles script"""
        behaviour_df, behaviour_matrix, df_articles, df_users = DataProcessor().generate_reading_behaviour_matrix()
        ModelFitter(behaviour_df, behaviour_matrix, df_articles, df_users).fit_model()


if __name__ == "__main__":
    CollaborativeFiltering().run()
