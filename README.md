# Sentiment Analysis for Amazon Review Dataset & Drug Dataset
![alt text](https://github.com/XiongjieDai/Sentiment-Analysis/blob/main/graphs/wordcloud.jpg?raw=true "Title or Description here")
## Introduction
Machine learning techniques have become increasingly popular and relevant to solve text and sentiment related problems in recent years. It has boosted performance on several tasks and significantly reduced the necessity for human efforts. For this project, we will focus on text classification, especially sentiment analysis, on several datasets. Since there are researches about sentiment analysis on _Amazon Review_ dataset, we will first use the same methods but R packages on the same dataset, and then replicate the methods on datasets other than the _Amazon Review_ dataset. By completing the project, we are trying to realize the following goals:
- Build four text sentiment classifiers on [Amazon Review](https://drive.google.com/drive/folders/14vWNcYX7ajd2YjPbf6Vi9cY35q3kq8ZE) dataset (`BoW`, `Word2Vec`, `GloVe`, `fastText`);
- Re-apply the above four classifiers on [Drug Reveiw dataset](https://drive.google.com/drive/folders/14vWNcYX7ajd2YjPbf6Vi9cY35q3kq8ZE);
- Compare different classifiers for datasets



## Group Member & Peer Evaluation

Rong Li: Data preprocessing, visualization, apply word2vec on Amazon review and Drug review dataset. (33.3%)

Xiongjie Dai: Data preprocessing, Word Cloud visualization, apply GloVe on Amazon review and Drug review dataset, and building Rmarkdown file. (33.3%)

Zixing Deng: BoW and FastText on Amazon review and Drug review dataset. (33.3%)

## Data

For the _Amazon Review_ dataset, we use the dataset constructed and made available by Zhang, Zhao and LeCun (2015). The dataset contains about 1,800,000 training samples and 200,000 testing samples with three attributes, which are classification labels (1 for negative reviews and 2 for positive reviews), the title of each review text, and the review text body. Due to the limit of computer computation ability, we pulled out the first 100,000 data samples and split 80% of the data into the training set and 20% of the data into the testing set.

For the Drug dataset, we downloaded from the UCI Machine Learning Repository. The dataset has 215,063 samples with 6 attributes, including drugName, condition, review, rating (1 to 10), date, usefulCount. Similar to _Amazon Review_ dataset, we split the whole dataset into training (80%) and testing (20%) datasets. In order to replicate the code, we categorized the dataset into two labels, where rating range from [1 to 4] are categorized as [1(negative)] and the rating range from [7 to 10] are categorized as [2(positive)]. We combined the text columns together and removed symbols that are not a number or a word. Moreover, we only retained the columns of rating and merged columns of review, name, condition as one text body attribute. Hence our drug dataset has the same format as _Amazon Review_.

## BoW approach with Naive Bayes

Bag of Words (BoW) method is widely used in NLP and computer vision fields. It takes the occurrence of each word in the text regardless of grammar and makes it into “bags” to characterize the text. To implement BoW method for our dataset, _Amazon Review_ and _Drug Review_, we first use `VCorpus` function and `DocumentTermMatrix` function in the `tm` package to convert text into a Document Term Matrix (DTM). By adjusting the built-in parameter in the `DocumentTermMatrix` function, we do not have to worry about cleaning the dataset with stop words. In order to make the model more precise, we removed words that do not occur in 99% of the documents by using `removeSparseTerms` function.

After finishing the process of BoW conversion, we can use the DTM to create word clouds for both positive and negative sentiment cases. And also to better interpret our word cloud, we simply use the two-sample t-test to better discriminant our words. Finally, we followed Chinnamgari (2019) and used the Naive Bayes sentiment classifier to perform predictions. Utilizing the `nb_sent_classifier` in the `e1071` package, we obtained the prediction results with approximate **81.19%** for _Amazon Review_ data and **74.77%** for _Drug Review_.

## Pretrained word2vec word embedding with Random Forest algorithm

`Word2vec` was developed by Mikolov et al. (2013) as a response to making the neural-network-based training of the embedding more efficient, and since then it has become the standard method for developing pretrained word embedding. The `softmaxreg` library in R offers pretrained `Word2vec` word embedding that can be used for building our sentiment analysis engine for the sentiment reviews data. The pretrained vector is built using the word2vec model, and it is based on the [Reuter_50_50 dataset UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Reuter_50_50).

After obtaining the word embedding, we calculated the review sentences embedding by taking the mean of all the word vectors of the words made up of the review sentences. Finally, the machine learning classification method is applied to the review sentence embeddings. In this problem, we used Random Forest algorithm to make classification and achieve an accuracy of **62.56%** on _Amazon Review_ and **70.99%** on _Drug Review_.

## GloVe word embedding with Random Forest algorithm

Pennington, Socher and Manning (2014) developed an extension of the `Word2vec` method called GloVe Vectors for Word Representation (`GloVe`) for efficiently learning word vectors. It combines the global statistics of matrix factorization techniques with local context-based learning in `Word2vec`. Also, unlike `Word2vec`, rather than using a window to define local context, `GloVe` constructs an explicit word context or word co-occurrence matrix using statistics across the whole text corpus. As an effect, the learning model yields generally better word embeddings.

The `text2vec` package in R has a `GloVe` implementation that we could use to train to obtain word embeddings from our own training corpus. Similar to the previous part, we used the `softmaxreg` library to obtain the mean word vector for each review. In this problem, we used the Random Forest algorithm to make classification and achieve an accuracy of **72.72%** on _Amazon Review_ and **74.96%** on _Drug Review_.

## FastText word embedding

`fastText` is also an extension of `Word2vec`, `fastTextR` package is used to reach more concise predictions for the analysis. Created and open-sourced by Facebook in 2016 (Mannes, 2016), `fastText` is a more powerful tool to classify text and learn word vector representation by breaking words into several character n-grams. `fastText` can construct the vector for a word from its character n-grams, even if it doesn’t appear in the training corpus; however, it is also time-consuming.

Before training the model, we convert the label in the dataset from “\\\1\\\” into “__label__1” in order to meet the format of the `fastText` algorithm. We also cleaned all multiple spaces in the text with a single space. Thereupon, we used `ft_train` function to train the model and `ft_control` to tune the hyperparameter for our two datasets. Our best accuracy for the fastText model is **86.49%** for the _Amazon Review_ and **78.69%** for the _Drug Review_.

## Conclusion & Disscusion

In conclusion, comparing all of our models after fine-tuning, the `fastText` model performs best on the _Amazon Review_ with **86.49%** of accuracy and the _Drug Review_ with **78.69%** of accuracy. Since words passed by the `fastText` model are represented as the sum of each word’s bag of character n-grams, `fastText` is much more efficient for dealing with large corpus and computing word embeddings for words unseen from the training set (Bojanowski et al., 2016). With such features, `fastText` can cope with typos and different word tenses accordingly without treating them as different words. For example, “helped” and “help” are two same words but only different from tenses. However, models other than `fastText` may treat them as two different words and assign the wrong labels. Therefore, using fastText can significantly boost performance.

From the entire scope, the `Word2vec` model performs with relatively low accuracy for both models (**62.56%** for Amazon Review and **70.99%** for Drug). One possible reason will be the existing words-only embedding in the `Word2vec` package. When encountering a sentence that its own embeddings cannot convert, the model will turn it into zero vectors, which will lose information and weaken the info of other word features. Thus, we may want to try to create and train word embedding on our own using CBOWand Continuous Skip-Gram in the future to see if any improvements can be made.

For future investigation, we can try to use BERT to better process the dataset and attempt more classification algorithms, such as XGBoost and AdaBoost, to classify the sentence embeddings. Moreover, as we only split our data into train and test datasets with 80-20, we may want to split the test set further into 10% of the validation dataset and 10% of the testing dataset so that we can select more fitting hyperparameters for the model and realize higher accuracy.
