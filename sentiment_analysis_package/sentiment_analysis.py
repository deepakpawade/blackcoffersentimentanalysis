import sys
import nltk
from nltk.corpus import stopwords
import re
import string
import copy
import os
from nltk.tokenize import word_tokenize

from sentiment_analysis_package import path_config
stopwords_dir = path_config.STOPWORDS_PATH
cols = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE',
       'POLARITY SCORE', 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',
       'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
       'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
       'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']

class sentiment_analysis:
    def __init__(self, dataframe, column):
        """
        dataframe = dataframe which has the text to be analyzed

        column = column name within the dataframe which has the text

        """
        self.dataframe = dataframe
        self.column = column
        self.stopwords = self._get_stopwords()
        self.pos_dict = self._get_positive_words(self.stopwords)
        self.neg_dict = self._get_negative_words(self.stopwords)
        self.output = self._get_analysis(copy.deepcopy(self.dataframe))

    def _get_stopwords(self):
        """
        Get all the stopwords from the StopWords folder
        """
        stopwords_files = ['StopWords_Auditor.txt',
                           'StopWords_Currencies.txt',
                           'StopWords_DatesandNumbers.txt',
                           'StopWords_Generic.txt',
                           'StopWords_GenericLong.txt',
                           'StopWords_Geographic.txt',
                           'StopWords_Names.txt']
        stopwords = set()
        for filename in stopwords_files:
            with open(stopwords_dir + '/'+filename, 'r', encoding='latin-1', errors='replace') as file:
                stopwords.update([word.strip() for word in file])
        # print(stopwords)
        return stopwords

    def _get_positive_words(self, stop_words):
        """
        Get all the positive words from the MasterDictionary folder
        """
        with open(path_config.MASTERDICTIONARY_PATH +'/positive-words.txt', 'r') as file:
            positive_words = [word.strip() for word in file.readlines()]

        # Remove stop words from the dictionaries
        pos_dict = {word for word in positive_words if word not in stop_words}
        return set(pos_dict)

    def _get_negative_words(self, stop_words):
        """
        Get all the negative words from the MasterDictionary folder
        """
        with open(path_config.MASTERDICTIONARY_PATH +'/negative-words.txt', 'r') as file:
            negative_words = [word.strip() for word in file.readlines()]

        neg_dict = {word for word in negative_words if word not in stop_words}
        return set(neg_dict)

    def _calculate_sentiment_scores(self, text, pos_dict, neg_dict):
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())

        # Calculate positive and negative scores
        pos_score = sum(1 for word in tokens if word in pos_dict)
        neg_score = sum(-1 for word in tokens if word in neg_dict) * -1

        # Calculate polarity score
        polarity_score = (pos_score - neg_score) / \
            (pos_score + neg_score + 0.000001)

        # Calculate subjectivity score
        subjectivity_score = (pos_score + abs(neg_score)
                              ) / (len(tokens) + 0.000001)

        return pos_score, neg_score, polarity_score, subjectivity_score

    def _clean_text(self, text):
        """
        This function cleans the given text by removing punctuations and stop words.
        """
        stop_words = set(stopwords.words('english'))
        cleaned_text = ""
        for word in nltk.word_tokenize(text):
            word = word.lower()
            if word not in stop_words and word not in string.punctuation:
                cleaned_text += word + " "
        return cleaned_text.strip()

    def _syllable_count(self, words):
        vowels = ['a', 'e', 'i', 'o', 'u']
        count = 0
        for word in words:
            if word[-2:] == 'es' or word[-2:] == 'ed':
                continue
            word_count = 0
            for char in word:
                if char.lower() in vowels:
                    word_count += 1
                count += word_count
        return count

    def _average_word_length(self, words):
        total_char = 0
        for word in words:
            total_char += len(word)
        return total_char/len(words)

    def _personal_pronouns(self, text):
        """
        returns number of personal pronouns in the provided text whilst avoing abbreviations
        """
        # first implemented using regex
        # pattern = r'\b(I|we|my|ours|us|We|My|Ours|Us)\b(?!\b[A-Z]{2}\b)'
        # matches = re.findall(pattern, text)
        # return len(matches)
        
        tokens = word_tokenize(text)
        # might need nltk.download('averaged_perceptron_tagger')

        pos_tags = nltk.pos_tag(tokens)

        pronouns = [word for word, pos in pos_tags if pos == 'PRP']
        return len(pronouns)
        

    def _initialize_cols(self,dataframe):
            dataframe['POSITIVE SCORE'] = 0
            dataframe['NEGATIVE SCORE'] = 0
            dataframe['POLARITY SCORE'] = 0
            dataframe['SUBJECTIVITY SCORE'] = 0
            dataframe['AVG SENTENCE LENGTH'] = 0
            dataframe['PERCENTAGE OF COMPLEX WORDS'] = 0
            dataframe['FOG INDEX'] = 0
            dataframe['AVG NUMBER OF WORDS PER SENTENCE'] = 0
            dataframe['COMPLEX WORD COUNT'] = 0
            dataframe['WORD COUNT'] = 0
            dataframe['SYLLABLE PER WORD'] = 0
            dataframe['PERSONAL PRONOUNS'] = 0
            dataframe['AVG WORD LENGTH'] = 0

    def _get_analysis(self, dataframe):
        """
        Requires : dataframe = dataframe which has the text to be analyzed

        Returns the provided dataframe with following variables and corresponding calculations for each row.
        1. All input variables in the dataframe
        2. POSITIVE SCORE
        3. NEGATIVE SCORE
        4. POLARITY SCORE
        5. SUBJECTIVITY SCORE
        6. AVG SENTENCE LENGTH
        7. PERCENTAGE OF COMPLEX WORDS
        8. FOG INDEX
        9. AVG NUMBER OF WORDS PER SENTENCE
        10. COMPLEX WORD COUNT
        11. WORD COUNT
        12. SYLLABLE PER WORD
        13. PERSONAL PRONOUNS
        14. AVG WORD LENGTH
        """
        self._initialize_cols(dataframe)
        for i, row in dataframe.iterrows():
            text = row[self.column]
            sentences_uncleaned = nltk.sent_tokenize(text)
            words_uncleaned = nltk.word_tokenize(text)
            cleaned_text = self._clean_text(text)
            sentences_cleaned = nltk.sent_tokenize(cleaned_text)
            words_cleaned = nltk.word_tokenize(cleaned_text)
            pos, neg, pol, sub = self._calculate_sentiment_scores(
                text, self.pos_dict, self.neg_dict)
            avg_sent_len = len(words_cleaned)/len(sentences_uncleaned)
            comp_word_count = sum(
                [1 for word in words_cleaned if len(word) > 2])
            per_comp_words = comp_word_count/len(words_cleaned)
            fog = 0.4 * (avg_sent_len + per_comp_words)
            avg_words_per_sent = len(words_cleaned)/len(sentences_uncleaned)
            syl_per_word = self._syllable_count(words_cleaned)
            per_pro = self._personal_pronouns(text)
            avg_word_len = self._average_word_length(words_cleaned)

            dataframe.at[i,'POSITIVE SCORE'] = pos
            dataframe.at[i,'NEGATIVE SCORE'] = neg
            dataframe.at[i,'POLARITY SCORE'] = pol
            dataframe.at[i,'SUBJECTIVITY SCORE'] = sub
            dataframe.at[i,'AVG SENTENCE LENGTH'] = avg_sent_len
            dataframe.at[i,'PERCENTAGE OF COMPLEX WORDS'] = per_comp_words
            dataframe.at[i,'FOG INDEX'] = fog
            dataframe.at[i,'AVG NUMBER OF WORDS PER SENTENCE'] = avg_words_per_sent
            dataframe.at[i,'WORD COUNT'] = len(words_cleaned)
            dataframe.at[i,'SYLLABLE PER WORD'] = syl_per_word
            dataframe.at[i,'PERSONAL PRONOUNS'] = per_pro
            dataframe.at[i,'AVG WORD LENGTH'] = avg_word_len
            dataframe.at[i,'COMPLEX WORD COUNT'] = comp_word_count

    
        # print(dataframe.columns)
        for col in dataframe.columns:
            if col not in cols:
                dataframe.drop(columns=[col],inplace=True)
                print(f'dropped {col}')
        dataframe = dataframe.reset_index(drop=True)
        dataframe = dataframe.set_index('URL_ID')

        return dataframe
