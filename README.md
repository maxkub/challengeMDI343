# challengeMDI343

## The final preprocessing of the dataset 

In that order:

#### - separating the dataset into two subsets

The subset is seprated using the feature 'SOURCE_CITED_AGE' = 'IMPUT' or 'CALC'.
Then each subset has its own preprocessing: the irrelevant features, the replacement of missing values with median values is different for each subset.


#### - removing features pointed as irrelevant
- The feature 'PRIORITY-MONTH' is removed: it is always the same date as 'BEGIN-MONTH' except when it has missing values.
- After RFECV from sklearn, with a 3-folds cross-validation on each subset, more features are pointed as irrelevant.

#### - dates
The dates are not useful as such, I keep only the year because decision trees and random forest showed that it is the most important feature.
I also created features describing the length (in days) beetween the three dates 'BEGIN-MONTH', 'FILING-MONTH', 'PUBLICATION-MONTH'.

#### - categorical values
Categorical values are encoded with label encoder from sklearn.

#### - missing data
The missing data are filled with the median value for each feature. This is done separately for each subset.


## The tested algorithms

- decision trees: poor results, I used it essentially to try to find important variables in the decision, to create new features.
- random forest: that gave results with a ROC_auc_score around 0.68. It was my first benchmark model.
- multilayer perceptron with tensorflow. The code I have written to use this library is given in the module data_science. It did not give really good results, 
but I did not try it with one-hot-encoder for the categorical variables...  
- Xgboost: that was my second benchmark model, with poorly optimized hyperparameters.


## Steps that improved the score

#### RFECV

To search for irrelvant features in the dataset.

#### Separating the dataset into two subsets.
When ploting histograms of each features, we see that some features have a very unbalanced distribution, for example: 'oecd_NB_BACKWARD_PL', 
so the learning algorithm would see a lot of 0 for this feature and few other values. As a result it could hard to train the algorithm to
use properly this feature. 

The idea is then 

## les traitements qui n'ont pas eu d'impact

#### feature engineering 

