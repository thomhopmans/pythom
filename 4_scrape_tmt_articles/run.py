import urllib

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


class ScrapeTMTArticles:
    """Create dataframe containing the links, titles, tags and content of all The Marketing Technologist articles."""
    URL_HOST = "http://www.themarketingtechnologist.co"

    def __init__(self):
        self.articles_df = pd.DataFrame(columns=['title', 'link', 'content_html', 'content_text', 'tags'])
        self.url_path = ""
        self.id = 1

    def run(self):
        """
        Run the Scrape TMT articles script
        :return:
        """
        self.get_overview_articles()
        self.get_content_articles()
        self.save_articles_df()

    def get_overview_articles(self):
        """

        :return:
        """
        # Continue until the last blog post page is reached.
        pagination_older_posts_link = ""
        while pagination_older_posts_link is not None:
            # Load content from URL
            page_url = self.URL_HOST + self.url_path
            content = urllib.urlopen(page_url).read()
            soup = BeautifulSoup(content)
            print(f"Extracting articles from {page_url}.")
            # Find all articles on page
            articles = soup.find_all("article", {"class": "post"})
            # Extract from each article on the page specific values such as the title, author, tags and link.
            for article in articles:
                self.extract_article_metrics(article)
            # Check if there is pagination link to older posts
            pagination_older_posts_link = self.check_for_older_posts_pagination_link(soup)
            # Next url path is the link to the older posts
            if pagination_older_posts_link is not None:
                self.url_path = pagination_older_posts_link
        # Output number of blog posts
        print(f"Number of blog post articles: {len(self.articles_df)}")

    def extract_article_metrics(self, article):
        """
        Get title, author, tags and link from article element.
        """
        article_title = article.find('a').getText()
        article_link = article.find('a').get("href")
        article_post_meta_links = article.find('footer', {"class": "post-meta"}).find_all("a")
        article_tags = []
        for post_meta_link in article_post_meta_links:
            # Check if /tag/ is in the link, o.w. it is not a tag-link but e.g. author
            if '/tag/' in post_meta_link.get('href'):
                article_tags.append(post_meta_link.getText())
        # Add article to dataframe
        self.articles_df.loc[self.id] = [article_title, article_link, "", "", article_tags]
        self.id += 1

    @staticmethod
    def check_for_older_posts_pagination_link(soup):
        """

        :param soup:
        :return:
        """
        pagination_older_posts_link = None
        pagination = soup.find("nav", {"class": "pagination"})
        # If there is pagination, check if there is an older posts link
        if pagination is not None:
            pagination_older_posts = pagination.find("a",  {"class": "older-posts"})
            # If there are older posts, get link of older posts page
            if pagination_older_posts is not None:
                pagination_older_posts_link = pagination_older_posts.get("href")
        return pagination_older_posts_link

    def get_content_articles(self):
        """
        Fill the dataframe with the content of all blog posts.
        """
        # Initialize Firefox browser for loading the blog posts. Note that we now switch to Selenium because we need
        # JavaScript to be executed on the page first to load the content.
        driver = webdriver.Firefox()

        # Iterate over all articles in the dataframe
        for index, row in self.articles_df.iterrows():
            # Load URL of blog post
            self.url_path = self.URL_HOST + row['link']
            print(f"Extracting article from {self.url_path}.")

            # Find post section on page
            driver.get(self.url_path)
            post_content_element = driver.find_element_by_class_name('post-content')
            article_content_html = post_content_element.get_attribute('innerHTML')
            article_content_text = post_content_element.text

            # Set content in articles dataframe
            self.articles_df.set_value(index, "content_html", article_content_html)
            self.articles_df.set_value(index, "content_text", article_content_text)

        # Close browser windows
        driver.quit()

    def save_articles_df(self):
        """
        Save articles_df to csv file
        """
        self.articles_df.to_csv("articles.csv", encoding='utf-8')


if __name__ == "__main__":
    ScrapeTMTArticles().run()
