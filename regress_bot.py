# -*- coding: utf-8 -*-

# IMPORTING PACKAGES ------------------------------------------------

import funcs # File that contains useful functions
import pandas as pd, tweepy, time, re, random
from datetime import date, datetime

from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# TWITTER KEYS AND "CONSTANT VARIABLES" -----------------------------

CONSUMER_KEY = 'XXX'
CONSUMER_SECRET = 'XXX'
ACCESS_KEY = 'XXX'
ACCESS_SECRET = 'XXX'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)

my_username = '@RegressML' # Username of the program tweet account
mark = 'Regress.'          # Like a signature - used in the end of a tweet
list_word = 'lista'        # Reserved word to a user ask for the companies that we already predict
sleep_time = 20            # Seconds between each requisition of mentions
predictions_time = (4, 0)  # (Hour, minute), time to tweet the predictions
link = 'https://finance.yahoo.com/quote/{}/history?p={}' # Link of the website that we get the data: Yahoo Finance
companies_file = 'companies.txt'
last_mention_id_file = 'last-mention-id.txt'
report_file = 'report.csv'

# Generic texts for tweeting
register_company_text = 'Nova ação para análise cadastrada por @{}:\n{} ;)\n\n{}'
company_not_found_text = 'Oi @{}!\nNão encontramos essa ação: {} :(\nVocê digitou certo?\n\n{}'
already_registered_company_text = 'Eai, @{}!\nJá tweetamos previsões de {} :D\n\n{}'
companies_list_text = 'Eai @{}!\nTweetamos previsões dessas ações: {}\n\n{}'
intro_tweets = [
    'Sobre {}, que fechou ontem em R$ {}, a gente tem umas previsões\n',
    '{} fechou ontem em R$ {}. Trouxemos nossas previsões pra você\n',
    'Se liga nas nossas previsões pra {}, que fechou ontem em R$ {}\n']
predictions_text = '{}\n{}\nFechamento de hoje\n {}: R$ {}\n {}: R$ {}\n {}: R$ {}'

df_length = 30
test_lines = 5
models = [
    [SGDRegressor,
    [['l1', 'l2', 'elasticnet'], (0.0001, 10), ['constant', 'optimal', 'adaptive']], 'SGD'],
    [Ridge,
    [(0.001, 10)], 'Ridge'],
    [LinearSVR,
    [(0.0001, 10)], 'SVR'],
    [KNeighborsRegressor,
    [(1, 5), ['uniform', 'distance']], 'KNN'],
    [RandomForestRegressor,
    [(1, 500)], 'RF'],
    [AdaBoostRegressor,
    [(1, 500), (0.01, 5)], 'Ada'],
    [MLPRegressor,
    [['identity', 'logistic', 'relu'], ['constant', 'invscaling', 'adaptive']], 'MLP']
    ]

# THE LOOP ----------------------------------------------------------

while True:
    # Wait some time to look for new mentions - Twitter has a limit of requests over time
    time.sleep(sleep_time)
    now = datetime.now()

    # SEARCHING FOR NEW MENTIONS AND REPLYING -----------------------

    last_read_tweet = funcs.read_file(last_mention_id_file)
    new_mentions = api.mentions_timeline(since_id = last_read_tweet) # Get the mentions since the last read tweet
    if len(new_mentions) == 0:
        print("\n{}\n- You don't have new mentions.".format(str(now)))
    else:
        # If there's new mentions, we write the id of the last one on the file, and searches for patterns to see
        # if the user is requesting a new company or the list of the companies already registered
        print("\n{}\n- You have {} new mentions!".format(now, len(new_mentions)))
        funcs.write_last_mention_id(new_mentions[0].id)
        for new_mention in new_mentions:
            api.create_favorite(new_mention.id)
            companies = funcs.read_file(companies_file)
            # If the new mention contains a regex (@[my_username] ".*") that is used to register new companies
            if funcs.contains_new_company(new_mention, my_username):
                tweeted_company = funcs.get_company_in_tweet(new_mention)
                if tweeted_company in companies:
                    funcs.reply_register_mention(api, new_mention, tweeted_company, mark, 'already registered', already_registered_company_text)
                else: # If the tweeted company is not registered, we need to see if this company exists
                    if funcs.company_exists(tweeted_company, link):
                        funcs.reply_register_mention(api, new_mention, tweeted_company, mark, 'new company', register_company_text)
                        funcs.register_company(tweeted_company, companies_file)
                    else:
                        funcs.reply_register_mention(api, new_mention, tweeted_company, mark, 'company not found', company_not_found_text)
            # If the user requested the list of the registered companies
            elif funcs.wants_list(new_mention, my_username, list_word):
                funcs.reply_list(api, new_mention, companies, mark, companies_list_text)

    # MODELS AND PREDICTIONS TWEETS ---------------------------------

    if not((now.hour == predictions_time[0]) & (now.minute == predictions_time[1]) & (date.today().weekday() in [0, 1, 2, 3, 4])):
        print("- It's not time to tweet predictions.")
    else:
        print("- It's time time tweet predictions!")
        companies = funcs.read_file(companies_file)
        for company in companies:
            df, last_close = funcs.get_data_from_web(link, company, df_length)

            X1_train, X1_test, y1_train, y1_test = funcs.get_train_test_data(df, 1, test_lines)
            best_models_1_day, hps, rmses, baseline_rmses = funcs.get_best_3_models(models, X1_train, X1_test, y1_train, y1_test, 1)
            p1 = funcs.get_predictions(best_models_1_day, 1, 10, df)
            funcs.update_report(models, report_file, company, p1, hps, rmses, baseline_rmses)

            funcs.tweet_predictions(api, predictions_text, intro_tweets, company, last_close, p1)
        time.sleep(60)
