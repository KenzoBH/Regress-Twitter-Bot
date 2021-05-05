# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
import random
from datetime import date

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize

def read_file(file_to_open):
    file = open(file_to_open, 'r')
    if file_to_open == 'last-mention-id.txt':
        read_lines = file.readlines()[0]
    elif file_to_open == 'companies.txt':
        read_lines = file.readlines()[0].split(',')
    file.close()
    return read_lines

def write_last_mention_id(mention_id):
    last_mention_id_file = open('last-mention-id.txt', 'w')
    last_mention_id_file.writelines(str(mention_id))
    last_mention_id_file.close()
    print('[Last mention id registered.]')

def contains_new_company(tweet, username):
    if re.search('{} ".*"'.format(username), tweet.text) == None:
        return False
    else:
        return True

def get_company_in_tweet(tweet):
    company_name = ((re.search('".*"', tweet.text).group())[1:-1]).upper()
    return company_name

def company_exists(company, link):
    company_link = link.format(company + '.SA', company + '.SA')
    print('Verifying the company. Searching on the internet...')
    try:
        pd.read_html(company_link)[0]
        return True
    except ValueError:
        return False

def reply_register_mention(api, new_mention, company, mark, condition, tweet_text):
    if condition == 'already registered':
        print('New mention contains a company that is already registered: {}.'.format(company))
    elif condition == 'new company':
        print('New mention contains a new company to register: {}.'.format(company))
    else:
        print("New mention contains a company that wasn't found: {}.".format(company))
    text_to_tweet = tweet_text.format(new_mention.user.screen_name, company, mark)
    api.update_status(text_to_tweet, in_reply_to_status_id = str(new_mention.id))
    print('-----\nNew tweet:\n{}\n-----'.format(text_to_tweet))

def register_company(company, companies_file):
    companies = read_file(companies_file)
    companies.append(company)
    file = open(companies_file, 'w')
    file.writelines(','.join(sorted(companies)))
    file.close()
    print('[New company registred: {}.]'.format(company))

def wants_list(tweet, username, list_word):
    if re.search('{} {}'.format(username, list_word), tweet.text) == None:
        return False
    else:
        print('New mention is requesting the companies list.')
        return True

def reply_list(api, tweet, companies, mark, companies_list_text):
    companies_list_as_str = ', '.join(companies)
    tweet_text = companies_list_text.format(tweet.user.screen_name, companies_list_as_str, mark)
    api.update_status(tweet_text, in_reply_to_status_id = str(tweet.id))
    print('-----\nNew tweet:\n{}\n-----'.format(tweet_text))

def get_data_from_web(link, company, df_length):
    print('Getting {} data from web...'.format(company))
    link_company = link.format(company + '.SA', company + '.SA')
    df = pd.read_html(link_company)[0].drop('Adj Close**', axis = 1)

    df = df[~df['Open'].str.contains('Dividend')]
    df = df[~df['Open'].str.contains('Split')]
    df = df[df['Volume'] != '-'].head(df_length)

    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open', 'High', 'Low', 'Close*', 'Volume']] = df[['Open', 'High', 'Low', 'Close*', 'Volume']].astype('float')

    df['Next Day Close'] = df['Close*'].shift(1)
    df['Next 5th Day Close'] = df['Close*'].shift(5)

    today_data = df.drop(['Next Day Close', 'Next 5th Day Close'], axis = 1).iloc[0]
    last_close = round(float(list(today_data)[4]), 2)
    print('Done!')
    return df, last_close

def get_train_test_data(df, days, test_lines):
    not_features = ['Date', 'Next Day Close', 'Next 5th Day Close']
    if days == 1:
        X_train = df.drop(not_features, axis = 1)[test_lines:]
        X_test = df.drop(not_features, axis = 1)[days:test_lines]
        y_train = df['Next Day Close'][test_lines:]
        y_test = df['Next Day Close'][1:test_lines]
    else:
        X_train = df.drop(not_features, axis = 1)[(test_lines + 5):]
        X_test = df.drop(not_features, axis = 1)[days:(test_lines + 5)]
        y_train = df['Next 5th Day Close'][(test_lines + 5):]
        y_test = df['Next 5th Day Close'][5:(test_lines + 5)]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def pick_model(model, params):
    if model == SGDRegressor:
        mdl = SGDRegressor(penalty = params[0], alpha = params[1], learning_rate = params[2], early_stopping = True, max_iter = 100)
    elif model == Ridge:
        mdl = Ridge(alpha = params[0], max_iter = 100)
    elif model == LinearSVR:
        mdl = LinearSVR(C = params[0], max_iter = 100)
    elif model == KNeighborsRegressor:
        mdl = KNeighborsRegressor(n_neighbors = params[0], weights = params[1])
    elif model == RandomForestRegressor:
        mdl = RandomForestRegressor(n_estimators = params[0])
    elif model == AdaBoostRegressor:
        mdl = AdaBoostRegressor(n_estimators = params[0], learning_rate = params[1])
    elif model == MLPRegressor:
        mdl = MLPRegressor(activation = params[0], learning_rate = params[1], early_stopping = True, solver = 'lbfgs')
    return mdl

def get_best_3_models(models, X_train, X_test, y_train, y_test, days):
    print('\nTraining models for {} day(s)...'.format(days))

    baseline = np.ones(len(y_test)) * y_train.iloc[0]
    baseline_rmse = mean_squared_error(y_test, baseline) ** 0.5
    print('Baseline RMSE: R$ {}'.format(round(baseline_rmse, 3)))

    hps = []
    rmses = []
    final_list = []
    for mdl_and_params in models:
        model = mdl_and_params[0]
        params = mdl_and_params[1]
        model_name = mdl_and_params[2]
        def train_model(params):
            mdl = pick_model(model, params)
            mdl.fit(X_train, y_train)
            mdl_p = mdl.predict(X_test)
            rmse = mean_squared_error(y_test, mdl_p) ** 0.5
            return rmse

        mdl_gp = gp_minimize(train_model, params, n_calls = 30, n_random_starts = 10)
        best_params = mdl_gp.x
        mdl_rmse = mdl_gp.fun

        best_mdl = pick_model(model, best_params)
        final_list.append([mdl_rmse, model_name, best_mdl])
        print('{} RMSE: R$ {}'.format(model_name, round(mdl_rmse, 4)))
        print('    {}'.format(best_params))
        hps.append(best_params) # contem os melhores hps de cada modelo
        rmses.append(mdl_rmse)
    best_3_models = sorted(final_list)[:3]
    baseline_rmses = [baseline_rmse for _ in range(len(models))]
    return best_3_models, hps, rmses, baseline_rmses

def get_predictions(models, days, test_lines, df):
    not_features = ['Date', 'Next Day Close', 'Next 5th Day Close']
    scaler = StandardScaler()
    if days == 1:
        X_train = df.drop(not_features, axis = 1)[test_lines:]
    else:
        X_train = df.drop(not_features, axis = 1)[(test_lines + 5):]
    scaler.fit(X_train)

    last_data = df.iloc[0].drop(not_features)
    print(last_data)
    last_data = scaler.transform([last_data])
    X_to_fit = df.drop(not_features, axis = 1)[days:]
    X_to_fit = scaler.transform(X_to_fit)

    if days == 1:
        y_to_fit = df['Next Day Close'][1:]
    else:
        y_to_fit = df['Next 5th Day Close'][5:]

    predictions = []
    for model in models:
        model[2].fit(X_to_fit, y_to_fit)
        prediction = model[2].predict(last_data)[0]
        predictions.append((model[1], round(prediction, 2)))
    return predictions

def update_report(models, report_file, company, p1, hps, rmses, baseline_rmses): # Function that updates the report with new predictions data for future analysis
    report = pd.read_csv(report_file, index_col = 0) # df to append new data

    date_vector = [date.today() for _ in range(len(models))]
    model_names = [model[2] for model in models]
    company_vector = [company for _ in range(len(models))]
    right_side = [np.nan for _ in range(len(models))]
    error = [np.nan for _ in range(len(models))]
    real_close = [np.nan for _ in range(len(models))]

    new_data = pd.DataFrame({'Date': date_vector,
                    'Model': model_names,
                    'Hps': hps,
                    'Model RMSE': rmses,
                    'Baseline RMSE': baseline_rmses,
                    'Company': company_vector,
                    'Right Side': right_side,
                    'Error': error,
                    'Real Close': real_close})
    new_data['Prediction'] = np.nan
    new_data['Top 3'] = 0
    for top in p1:
        for i in range(len(new_data['Model'])):
            if new_data['Model'][i] == top[0]:
                new_data['Prediction'][i] = top[1]
                new_data['Top 3'][i] = 1
        
    new_report = pd.concat([report, new_data], ignore_index = True)
    new_report.to_csv(report_file)
    print('Report updated!\n')
    
def tweet_predictions(api, predictions_text, intro_tweets, company, last_close, p1):
    prediction_text = predictions_text.format(date.today(), random.choice(intro_tweets).format(company, last_close),
    p1[0][0], p1[0][1], p1[1][0], p1[1][1], p1[2][0], p1[2][1])
    print('-----\nNew tweet:\n{}\n-----'.format(prediction_text))
    api.update_status(prediction_text)
