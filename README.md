# challengeMDI343
All the created functions are in *utils/data_science.py*. All scores are evaluated with the roc_auc_score from sklearn, on a cross validation subset of the data representing 20% of the whole dataset.

### Preprocessing of the dataset
This is the preprocessing applied to find the best score. The following operations are applied, in that order:

##### 1. Separating the dataset into two subsets

The subset is separated using the feature SOURCE_CITED_AGE = IMPUT or CALC. More details will be given in the following.
Then each subset has its own preprocessing: the irrelevant features, the replacement of missing values with median values is different for each subset.


##### 2. Removing features pointed as irrelevant
- The feature PRIORITY-MONTH is removed: it is always the same date as BEGIN-MONTH except when it has missing values.
- After RFECV from sklearn, with a 3-folds cross-validation on each subset, more features are pointed as irrelevant.

##### 3. Dates
The dates are not useful as such, I keep only the year because decision trees and random forest showed that it is the most important feature.
I also created features describing the length (in days) between the three dates BEGIN-MONTH, FILING-MONTH, PUBLICATION-MONTH.
The original datetime type features are removed.

##### 4. Categorical values
Categorical values are encoded with label encoder from sklearn.

##### 5. Missing data
The missing data are filled with the median value for each feature. This is done separately for each subset.


### The tested algorithms

- **Decision trees:** very poor results, I used it essentially to try to find important variables in the decision, to create new features.
- **Logistic regression:** very poor results...
- **Random forest:** that gave results with a ROC_auc_score around 0.69. It was my first benchmark model, with the preprocessings described above.
- **Multilayer perceptron** with tensorflow. The code I have written to use this library is given in the module data_science. It did not give better results than random forest,
but I did not try it with one-hot-encoder for the categorical variables...
- **Xgboost:** that was my second benchmark model, although with poorly optimized hyperparameters, and with the same preprocessings as above.
- **Hyperopt** + Xgboost: the best results.


### Steps that improved the score

##### 1. Simplifying the preprocessings and the model, then, building with a method

This was, by far, the most efficient step to improve the score.

I started with some complicated preprocessings, with a lot of new features coming from the one-hot-encoding although I removed the features
containing many categories, and/or with some engineered features and scores. Even with a random forest with a *gridsearchCV*, I could not do better than
a score of ~0.64.

I was also working with a multilayer perceptron, I tested it from 1 layer with 2 neurons, up-to 3 layers with 150 neurons on each layers. It did not gave results above
~0.67 (which was obtained with networks having 1 hidden layer with 15-20 neurons).

So I started, to apply a more methodological method: apply very few preprocessings and test a good "off-the-shelf" algorithm (I tested with random forest).
This becomes a benchmark model. Then try to beat the benchmark by testing one new preprocessing, if it works keep the preprocessing, if it does not
work try something else. 


##### 2. RFECV

RFECV from sklearn was used to search for irrelevant features in the dataset.

##### 3. Separating the dataset into two subsets

When plotting histograms of each features, we see that some features have a very unbalanced distribution, for example: oecd_NB_BACKWARD_PL,
so the learning algorithm would see a lot of 0 for this feature and few other values. As a result it could be hard to train the algorithm to
use properly this feature.

The idea is then to look for a feature that could split the dataset in two subsets, under the constraints that:
- The average "distance" between the subsets must be the largest. For example, with a binary variables V, we would like one subset to take all the zeros of V,
and the other subset to take all the ones. So, we would like The average (or median) value of V in subset 1 to be close to zero, and in the subset 2 to be close
to one. The larger the difference between those two average, the better is our split. This difference is computed for each feature, the results are stored in a vector.
The final "distance" measurement between the two subsets is the norm of that vector.
- The subsets must have approximately the same size (if one is too small, it is hard to train a good model on it). This can be measured via a two-class
entropy.
- The overall distribution between the 4 categories (VARIABLE_CIBLE=0 or 1 in subset 1, and  VARIABLE_CIBLE=0 or 1 in subset 2) must be balanced.
This can be measured via a 4-class entropy.

I wrote the function *data_split_scores()* to give a score (which is the product of the three quantity described above) to each binary categorical variable,
to find the best splitting variable. I only looked among binary variables at first and found good results, so I did not tried more complicated variables and split
scores.

The resulting sorted scores of the function *data_split_scores()*, are at the entry [63] in *archive/challenge2.ipynb*.
The binary having the highest score is SOURCE_CITED_AGE, ex-aequo with SOURCE_IDX_ORI because these two features are always equal.
When plotting the distribution of the features in each subset, we can see that the very unbalanced
distributions that we found in the whole dataset, are split between the two subsets.

##### 4. Optimizing Xgboost model

This was done using hyperopt, on each subset. The score improved from ~0.70 to ~0.71 .
