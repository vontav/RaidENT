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


