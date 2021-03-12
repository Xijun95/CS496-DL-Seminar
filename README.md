# CS496-DL-Seminar

The course final delivery for replicating the paper: "Deep architectures for long-term stock price prediction with a heuristic-based strategy for trading simulations".

## Description of the paper

Stock price prediction is a popular yet challenging task and deep learning provides the means to conduct the mining for the different patterns that trigger its dynamic movement. In this paper, the authors first build and train the deep learning model to predict the close price of the stock, then they propose a trading strategy based on the prediction, whose decision to buy or sell depend on two different thresholds. Finally, a hill climbing approach selects the optimal values for these parameters.

## Problem Statement

Based on the paper's description, the problem can be separated into three steps:

* Build the **Deep Learning model** to predict the **close price**. 
* Propose the **trading strategy**, whose decision to buy or sell depends on **two different thresholds**.
* A **hill climbing** approach selects the optimal values for these thresholds. 

(For the first step, the authors tried a CNN and LSTM model to fulfill the same goal (predicting the close price), and based on the final results showed in the paper, the prediction made by the LSTM model provides a larger gain during the trading process, and since this is a one person team, in this blog, I mainly choose to use the LSTM-based model to make the stock close price prediction.)

## Data 

The Datasets used in the paper is text-based type, a sample of the dataset is shown below:


## Relevance
## Methodologies
## Codes
## Results
## Commands
