# Stock-Binary-Classification-LSTM
 Uses an LSTM to predict the next days stock movement based on sequence of previous days.  I used this project to gain expierience working with LSTM's and time series data.
 
 ![sp500](https://github.com/dgleaso/Stock-Binary-Classification-LSTM/blob/main/images/sp500-historical.png) ![cm-train](https://github.com/dgleaso/Stock-Binary-Classification-LSTM/blob/main/images/cm_train.png)

The python code here trains an LSTM, a type of recurrent neural network suited for sequences of data, to predict wether the S&P500 index price will move up or down based on the data of previous days.  The reason for making this a binary classification problem, rather than a regression problem (ie trying to predict the stock price), is that the stock price is highly autocorrelated.  This means it has high correlation with a delayed version of itself, meaning the best prediction of the stock price is the previous days stock price, which is what the LSTM would learn.  Changing the task to binary classification removes this problem.

![sp500](https://github.com/dgleaso/Stock-Binary-Classification-LSTM/blob/main/images/cm_train.png) ![cm-train](https://github.com/dgleaso/Stock-Binary-Classification-LSTM/blob/main/images/cm_test.png)

![sp500](https://github.com/dgleaso/Stock-Binary-Classification-LSTM/blob/main/images/acc_train.png) ![cm-train](https://github.com/dgleaso/Stock-Binary-Classification-LSTM/blob/main/images/acc_test.png)

As you can see in the above figures, the LSTM learns to classify quite well on the training data, however it does not generalize well to the testing data.  One problem is that the dataset is unbalanced as there are sigificantly more days where the stock price goes down rather than up.  In training, I used PyTorch's Binary cross entropy loss pos_weight parameter to counteract this, by increasing the weight assigned to positive datapoints proportionately to the unbalance in the dataset.  This allowed it to learn well on the training data, however it still struggles on the unseen testing data.  Perhaps more methods of regularization, such as dropout, would improve the generalizability of the model.
