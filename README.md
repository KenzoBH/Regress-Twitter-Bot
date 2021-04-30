<h1 align="center">Twitter Bot for Machine Learning: Stock Predictions</h1>   

![](https://github.com/KenzoBH/Data-Science/blob/main/Images/Regress.jpg)

# Overview   

This project is about publishing (or tweeting) forecasts of 7 models on stocks in the financial market everyday. Users can see the predictions and ask for predictions of the companies they want.     
You can see the portuguese project page here, in my website: [link](https://kenzobh.github.io/projects/Regress-Twitter-Bot.html). There, I discussed over the project in a simpler way.   
Technologies and packages used:
- Web Scraping: pandas
- Machine Learning Models: sklearn
- Twitter Bot: Tweepy
- Deploy: PythonAnywhere

You can see the final project [here](https://twitter.com/RegressML) (the Twitter account of Regress).

## Files

- [`regress_bot.py`](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/regress_bot.py) and [`funcs.py`](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/funcs.py): Python program that is running in [PythonAnywhere](https://www.pythonanywhere.com/), and contains the bot code and the models training. `funcs.py` is a file that contais functions that I used in the `regress_bot.py` program.
- [`companies_txt`](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/companies.txt) and [`last-mention-id.txt`](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/last-mention-id.txt): Example of text files used by the Regress Bot to store the companies that it predicts and tweets, and the last tweet that mentioned it's account (better explained on `regress_bot.py` logic and comments).

## Methods

The program train 7 models everyday, tuning their hyperparameters. They are: Stochastic Gradient Descent, Ridge Regression, Linear Support Vector Regressor, K-Nearest Neighbors, Random Forest, Ada Boost and Neural Networks. The models are trained with the last 30 days, and tested with the last 5 - the best 3 are chosen (based on their RMSE, Root Mean Squared Error) to tweet the predictions. As I observed, Linear SVR and SGD are the best ones.

As the data is a time series, the ideal would train models like Arima or Prophet, from Facebook. I chose to use more "classic" models, because I wanted to see how these models  would perform. As the next step, you could see how time series models would predict. Furthermore, as the program is hosted in a free platform, PythonAnywhere, it becomes impracticable train models with a lot of data - that's why I opted for 30 days (and using more days - I got to use 5 years - to train the models increased their RMSE, that's interesting). Maybe, with more recent data, the models learn stronger relations between the features and the label, one the recent data reflects better the today data.   
Maybe, training a MLP (Multi-Layer Perceptron) in a small dataset is disproportionate - but I wanted to see how it would perform -, however, it seems to be a good model.

More details of the models training and selection is in `regress_bot.py`and `funcs.py` files. They are trained everyday with new data, that is scraped off the [Yahoo Finance website](https://finance.yahoo.com/), and their predictions are shown on the [@RegressML](https://twitter.com/RegressML) account on Twitter. That simple.

-------------------------

<p align="center">My Data Science portfolio: <a href="https://github.com/KenzoBH/Data-Science">link</a><br>My LinkedIn: <a href="https://www.linkedin.com/in/bruno-kenzo/">link</a></p>
<p align="center">Bruno Kenzo, 17 yo.</p>


