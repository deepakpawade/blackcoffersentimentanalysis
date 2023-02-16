from sentiment_analysis_package.sentiment_analysis import sentiment_analysis
from sentiment_analysis_package.url_extractor import url_extractor
import pandas as pd
import copy
df =  pd.read_excel('./blackcoffer_sentiment_analysis/Input.xlsx')
df_out_analyzed = sentiment_analysis(df,'content')
df_out_analyzed.output.to_excel('./blackcoffer_sentiment_analysis/output.xlsx')
