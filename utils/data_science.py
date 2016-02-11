
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import operator


def import_data(train_path, test_path, target, na_values=None):
    # Import
    test = pd.read_csv(test_path, sep=';', na_values=na_values)
    test[target] = 'UNKNOWN'
    test['index_origin'] = test.index.tolist()

    train = pd.read_csv(train_path, sep=';', na_values=na_values)
    train['index_origin'] = -1

    piv_train = train.shape[0]

    # Creating a DataFrame with train+test data
    df_all = pd.concat((train, test), axis=0, ignore_index=True)

    return df_all

def sort_dict(dict, reverse=True):
    """
    sorting dictionnaries by values
    """
    return sorted(dict.items(), key=operator.itemgetter(1), reverse=reverse)


def value_counts(df, look_for=None):
    for col in df.columns:
        print(col)
        vals = df[col].value_counts(normalize=True, dropna=False).reset_index()
        print(vals)
        print()
        if look_for == None:
            print(vals[vals['index'].isnull()])
        else:
            print(vals[vals['index'] == look_for])
        print('-------------------------------------')


class Used_features():

    def __init__(self, target):
        self.target = target

    def fit_transform(self, df, except_cols=None):
        self.columns_Y = [col for col in df.columns if len(
            df[col].unique()) >= 2 or col == except_cols]
        self.columns_X = self.columns_Y.copy()
        self.columns_X.append(self.target)
        return df[self.columns_Y]

    def transform(self, df):
        return df[self.columns_X]

#-------------------------------------------------------------------------------------------------


class Preprocessings():

    def __init__(self, date_columns=None, cols_toDrop=None):
        self.dateCols = date_columns
        self.colsToDrop = cols_toDrop

        self.numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    def drop_cols(self, df, cols):
        return df.drop(cols, axis=1)

    def datetime_processings(self, df, format=None):
        # converting string dates in dateTime format :
        df[self.dateCols] = df[self.dateCols].apply(lambda col: pd.to_datetime(col, format=format))

        # PRIORITY_MONTH = BEGIN_MONTH :
        df = df.drop(['PRIORITY_MONTH'], axis=1)

        # creating features of duration between dates :
        df['filing-begin'] = (df.FILING_MONTH - df.BEGIN_MONTH).dt.days
        df['pub-filing'] = (df.PUBLICATION_MONTH - df.FILING_MONTH).dt.days
        df['pub_year'] = df.PUBLICATION_MONTH.dt.year
        df = df.drop(['FILING_MONTH', 'PUBLICATION_MONTH', 'BEGIN_MONTH'], axis=1)

        df = df.drop('cited_nmiss', axis=1)

        return df

    def cat_to_codes(self, df):
        """
        converting categorical data into numerical codes :
        NaN's will be replaced by -1
        And replacing -1 by the meadian value of each categorical column
        """

        # non numeric columns to be encoded
        self.non_num_cols = df.select_dtypes(
            exclude=self.numerics).columns.difference(self.dateCols)

        # numeric columns
        self.num_cols = df.select_dtypes(include=self.numerics).columns

        print(self.non_num_cols)

        df[self.non_num_cols] = df[self.non_num_cols].apply(
            lambda col: col.astype('category').cat.codes)

        df[self.non_num_cols] = df[self.non_num_cols].replace(-1, df[self.non_num_cols].median())

        return df

    def create_score(self, df, columns, target, score_name):
        df_score = df.groupby(columns).mean().add_suffix('_mean').reset_index()
        columns.append(target + '_mean')
        df_score = df_score[columns]
        df_score = df_score.rename(columns={target + '_mean': score_name})

        columns.remove(target + '_mean')
        df = pd.merge(df, df_score, on=columns, how='left')

        return df

    def reducing(self, df, cols=None):
        """
        reducing the continous features to a distribution with mean=0 std=1
        """
        if cols == None:
            columns = self.num_cols
        else:
            columns = cols

        df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()

        return df

    def oneHot_encoder(self, df, except_cols=None):

        to_encode = self.non_num_cols.tolist()
        if except_cols != None:
            to_encode.remove(except_cols)

        print('drop', to_encode)

        onehotencoder = OneHotEncoder(sparse=False)
        temp = onehotencoder.fit_transform(df[to_encode].values)

        cat_cols = []
        for j in range(len(to_encode)):
            for i in range(onehotencoder.feature_indices_[j + 1] - onehotencoder.feature_indices_[j]):
                cat_cols.append(to_encode[j] + '_' + str(i))

        df_temp = pd.DataFrame(temp, columns=cat_cols)

        df = pd.concat([df, df_temp], axis=1, join='inner')
        df = df.drop(to_encode, axis=1)

        return df


#--------------------------------------------------------------------------------------------------


def re_split(df, target, split_value=None):
    """
    When eval data set is imported, you need to create a column 'target' and filling it with
    a value that is not in the train data set. By doing so, after cat_to_codes the eval set is
    identified by the largest value in categorical 'target'
    """
    if split_value == None:
        unknown_target = df[target].unique().max()
    else:
        unknown_target = split_value

    df_train = df[df[target] != unknown_target]
    df_eval = df[df[target] == unknown_target]
    df_eval = df_eval.drop(target, axis=1)

    return df_train, df_eval


def data_split_scores(df, categorical_cols, print_results=10, numberOfSubgroups=4):
    """
    function to test if a binary class feature can be used to separate
    the dataset into two sub-datasets with different feature behavior.
    In order to train one different model on each sub-dataset.
    """
    Ltot = len(df)

    scores = {}
    for col in categorical_cols:
        groups = df.groupby(col).median().reset_index()
        groups = groups.transpose()
        groups['diff'] = groups[1] - groups[0]

        norm = np.linalg.norm(groups['diff'].values)

        p0 = df[col].value_counts()[0] / Ltot
        p1 = df[col].value_counts()[1] / Ltot
        entropy = -p0 * math.log10(p0) - p1 * math.log10(p1)

        groups_count = df.groupby((col, 'VARIABLE_CIBLE')).count().reset_index().iloc[:, 3].values
        groups_count = np.asarray(groups_count)

        groups_proba = - groups_count / Ltot * np.log10(groups_count / Ltot)
        tot_entropy = sum(groups_proba)

        score = norm * entropy * tot_entropy

        if len(groups_count) == numberOfSubgroups:
            scores[col] = score

    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

    if print_results != None:
        i = 0
        for k, v in scores:
            if i >= print_results:
                break
            print('%30a  %5.3f' % (k, v))
            i += 1

    return scores
