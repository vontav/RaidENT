# RaidENT


<h1>Usage</h1>
<d>Clone the repository into a folder you have access to. There are a couple main scripts that will guide you to the most sucess.</d>
<ol>
  <li>the data you will be using to train your own model is collected through phemex exchange. there will be 4 csv files created, each for different time intrevals. (1minute)(5minute)(15minute)(1hour) ALL YOU HAVE TO DO TO START COLLECTING DATA IS CD INTO THE PROJECT FILES FOLDER AND RUN THE DATA_COLLECTOR.PY SCRIPT USING PYTHON OF COURSE</li>
    <li>when you are happy with how much data you have you can train a model using the different scripts provided. the mechanics are in the title of the script and use different strategies to achieve the overall goal of predicting bitcoins USD value over a given intreval of time.</li>
  <li>Finally when you have trained up your new prediction model, you want to make sure the main.py file is adjusted so that the model it is using is the correct directory to the saved model. this would be on line 8</li>
  <li>There are alot of developmental features that currently do not have much meaning. For instance the mexcUSDTBTCPERP.py is going to build on the idea of using the futures market to better predict bitcoins value over time.</li>
</ol>

















---------------------------------------------------------------------------------------------------------------------------------------------------
key features of this model:

Long Short-Term Memory (LSTM) layers are used to capture temporal dependencies in the sequential data

Convolutional 1D layers apply convolutions to the data, helping to capture local patterns

The attention mechanism allows the model to focus on relevant features from both the LSTM and Conv1D branches

Outputs from different layers are concatenated and processed further to extract high-level patterns

Dense layers process the concatenated features and produce the final prediction

The model is compiled using the Adam optimizer and mean squared error (MSE) loss function.

---------------------------------------------------------------------------------------------------------------------------------------------------




Last Update: August 25th 2023

----------

-soft update was done to the data collector to begin splitting into different time intrevals. 
-prime2.0 training model added aswell. 

----------

working on:
---------------------------
-different time interval models
-prediction aggregation
-batch data
-new data sources (purchase orders/sell orders)
-hyperparameter tuning
-visualization
-ensemble methods
-top secret
---------------------------


