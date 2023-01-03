<h1 align="center">Twitter Bot for Machine Learning: Stock Predictions</h1>   

![](https://github.com/KenzoBH/Data-Science/blob/main/Images/Regress.jpg)

# Overview   

This project is about publishing (or tweeting) forecasts of 7 models on stocks in the financial market everyday. Users can see the predictions and ask for predictions of the companies they want.     
Some of the main technologies and packages used:
- **Web Scraping** using [pandas](https://pandas.pydata.org/)
- **Machine Learning** with [sklearn](https://scikit-learn.org/stable/) and [scikit-optimze](https://scikit-optimize.github.io/stable/)
- **Twitter Bot** on [Tweepy API](https://www.tweepy.org/)
- **Deploy**: I'm looking for another online hosting platform

You can see the final project [here](https://twitter.com/RegressML), the Twitter account of Regress.

# Files

- **[regress_bot.py](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/regress_bot.py) and [funcs.py](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/funcs.py):** Python program that is running in [PythonAnywhere](https://www.pythonanywhere.com/), and contains the bot code and the models training. `funcs.py` is a file that contais functions that I used in the `regress_bot.py` program.
- **[companies.txt](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/companies.txt) and [last-mention-id.txt](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/last-mention-id.txt):** Example of text files used by the Regress Bot to store the companies that it predicts and tweets, and the last tweet that mentioned it's account.
- **[report.csv](https://github.com/KenzoBH/Regress-Twitter-Bot/blob/main/report.csv):** Example of a report that's updated everyday by Regress Bot.

# Methods

The program train 7 models everyday, tuning their hyperparameters. They are: Stochastic Gradient Descent, Ridge Regression, Linear Support Vector Regressor, K-Nearest Neighbors, Random Forest, Ada Boost and MLP. The models are trained with the last 30 days, and tested with the last 5 - the best 3 are chosen (based on their RMSE, Root Mean Squared Error) to tweet the predictions. As I observed, Linear SVR and SGD are the best ones.

<div align="center">
  
| Model | Tuned Hyperparameters |
| :---: | :-------------------: |
| [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html?highlight=sgd#sklearn.linear_model.SGDRegressor) | Penalty, Alpha and Learning Rate |
| [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge) | Regularization |
| [Linear Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html?highlight=linearsvr#sklearn.svm.LinearSVR) | Regularization |
| [Regression based on k-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html?highlight=neighbors#sklearn.neighbors.KNeighborsRegressor) | Number of Neighbors and Weights |
| [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest#sklearn.ensemble.RandomForestRegressor) | Number of Trees |
| [Ada Boost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html?highlight=ada%20boost#sklearn.ensemble.AdaBoostRegressor) | Number of Estimators and Learning Rate |
| [Multi-layer Perceptron Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlp#sklearn.neural_network.MLPRegressor) | Activation Function and Learning Rate | 
  
</div>

As the data is a time series, the ideal would train models like [Arima](https://pt.wikipedia.org/wiki/ARIMA#:~:text=Em%20estat%C3%ADstica%20e%20econometria%2C%20particularmente,de%20m%C3%A9dias%20m%C3%B3veis%20(ARMA).) or [Prophet, from Facebook](https://facebook.github.io/prophet/).   
I chose to use more "classic" models, because I wanted to see how these models would perform. As the next step, you could see how time series models would predict. Furthermore, as the program is hosted in a free platform, PythonAnywhere, it becomes impracticable train models with a lot of data - that's why I opted for 30 days (and using more days - I got to use 5 years - to train the models increased their RMSE, that's interesting). Maybe, with more recent data, the models learn stronger relations between the features and the label, one the recent data reflects better the today data.   
Maybe, training a MLP (Multi-Layer Perceptron) in a small dataset is disproportionate - but I wanted to see how it would perform -, however, it seems to be a good model.

More details of the models training and selection is in [`regress_bot.py`](https://github.com/KenzoBH/Data-Science/blob/main/Twitter_ML/regress_bot.py) and [`funcs.py`](https://github.com/KenzoBH/Regress-Twitter-Bot/blob/main/funcs.py) files. They are trained everyday with new data, that is scraped off the [Yahoo Finance website](https://finance.yahoo.com/), the hyperparameters are tuned with skicit-optimize, and their predictions are shown on the [@RegressML](https://twitter.com/RegressML) account on Twitter.

-------------------------

<p align="center">My Data Science portfolio: <a href="https://github.com/KenzoBH/Data-Science">link</a><br>My LinkedIn: <a href="https://www.linkedin.com/in/bruno-kenzo/">link</a></p>
<p align="center">Bruno Kenzo, 18 yo.</p>


