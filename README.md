# Twitter-Sentiment-Classification-Using-Distant-Supervision
Twitter Sentiment Classification by applying several Machine Learning Classifiers and Artificial Neural Networks and using emoticons as nosiy lables.
Based on a [paper](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf) by Stanford University
# Datasets
* [English Tweets Dataset](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
* [Arabic Tweets Dataset](https://drive.google.com/file/d/1TNaUswMaVMhkWTEdNpeRBVwbizi6uKyZ/view?usp=sharing)
# Approcah 
* using different machine learning classifiers and feature extractors as well as Artificial Neural Networks (ANN). 
* The machine learning classifiers are Logistic Regression, Naive Bayes,, Multinomial NB, Ridge Classifier, Passive-Aggressive Classifier and Support Vector Machines (SVM). 
* The Artificial Neural Network is used along with Tfidf vectorizer 
* The feature extractors are unigrams, bigrams and trigrams. 
# Dataset Description and Preprocessing
## Description
Dataset has 1.6 million entries, with no null entries,the training set has no neutral class.50% of the data is with negative label, and another 50% with positive label.
The information on each field of the data set is:
0 — the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)  
1 — the id of the tweet (2087)  
2 — the date of the tweet (Sat May 16 23:58:44 UTC 2009)  
3 — the query (lyx). If there is no query, then this value is NO_QUERY.  
4 — the user that tweeted (robotickilldozr)  
5 — the text of the tweet (Lyx is cool)  

## Preprocessing
 * These Emoticons are stripped off: :) : ) :-) :D =) :( : ( :-(
 * Tweets containing both positive and negative emoticons are removed. 
 * Retweets are removed.
 * Tweets with “:P” are removed. 
 * Repeated tweets are removed.
 * Converting  HTML encoding to text
 * Replacing any url with class URL
 * Replacing any @username with class USERNAME
 * Striping  repeated chars. For example “Huuuuugry !” becomes “Huungry !”
 * Replacing #hashtag with hashtag
 * Removing Numbers
# Feature Extraction 
Two feature extraction methods are used : count vectorizer and TFIDF vectorizer, using different “n-grams”  (unigrams, bigrams and trigrams) with and without English stop words in dataset.
# Machine Learning Model 
## Classifiers 
1. Logistic Regression 
2. Naive Bayes 
3. Multinomial NB 
4. Ridge Classifier 
5. Passive-Aggressive Classifier 
6. Support Vector Machines (SVM) 
##  Artificial Neural Network
The structure of NN model has **100,000** nodes in the input layer, then **64** nodes in a hidden layer with Relu activation function applied, then finally one output layer with sigmoid activation function applied using **20%** drop out of hidden layer with shuffling data for each epoch.
# Results
* using emoticons as noisy labels for training data is an effective way to perform distant supervised learning.
* Logistic regression achieve highest accuracy of **82.73%** for classifying sentiment.
* Neural Network failed to outperform logistic regression in terms of validation.This might be due to the high dimensionality and sparse characteristics of the textual data.




