# RaidENT
---------------------------------------------------------------------------------------------------------------------------------------------------
key features of this model:

Long Short-Term Memory (LSTM) layers are used to capture temporal dependencies in the sequential data

Convolutional 1D layers apply convolutions to the data, helping to capture local patterns

The attention mechanism allows the model to focus on relevant features from both the LSTM and Conv1D branches

Outputs from different layers are concatenated and processed further to extract high-level patterns

Dense layers process the concatenated features and produce the final prediction

The model is compiled using the Adam optimizer and mean squared error (MSE) loss function.

---------------------------------------------------------------------------------------------------------------------------------------------------


trying to predict bitcoin every 5 mins. Different models included. bad data set though and im collecting it every 5 mins. wish i had a live set, also more data besides just btc price. ensemble methods would be benificial. 


Last Update: August 25th 2023

working on:
-different time interval models
-prediction aggregation
-batch data
-new data sources (purchase orders/sell orders)
-hyperparameter tuning
-visualization
-ensemble methods
-top secret


