import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import copy

import pandas as pd
import requests
from bs4 import BeautifulSoup


class url_extractor:
    def __init__(self, dataframe, column):
        self.dataframe = dataframe
        self.column = column
        self.output = self._dataframe_itr(copy.deepcopy(self.dataframe))


    def _extract_content(self, url):
        """
        Extracts contents from the provided url
        """

        # send an HTTP GET request to the website
        response = requests.get(url)

        # parse the HTML content of the website using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # if the page returns 404
        if soup.find('div', class_='td-404-title'):
            print(f'{url} 404 Error: Page not found')
            return None

        # extract the content of the article
        content = ""
        try:
            for p in soup.find("div", {"class": "td-post-content"}).find_all("p"):
                # content += p.text.strip() + "\n"
                content += p.text.strip() + " "
        except Exception as e:
            print(f'content : {e}: {url}')
        return content

    def _dataframe_itr(self, dataframe):
        """
        Iterates through the dataframe and adds extracted data to the corresponding rows with links
        """
        dataframe['content'] = ''

        # iterate over the rows of the dataframe and extract the content of each link
        # using df.apply() would work efficiently
        for i, row in dataframe.iterrows():
            try:
                link = row['URL']
                content = self._extract_content(link)
                dataframe.at[i, 'content'] = content
            except Exception as e:
                print(f'content : {e}: {link}')
        dataframe = dataframe[dataframe['content'] != ""]
        dataframe = dataframe.dropna()
        return dataframe