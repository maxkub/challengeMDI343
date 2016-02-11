
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

#-----------------------------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------------------------



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


#-----------------------------------------------------------------------------------------------------------


def data_split_scores(df, categorical_cols, print_results=10, numberOfSubgroups=4):
    """
    function to test if a binary class feature can be used to separate
    the dataset into two sub-datasets with different feature behavior.
    In order to train one different model on each sub-dataset.
    """
    Ltot = len(df)

    scores = {}
    for col in categorical_cols:

        # Computing a distance between two subsets:
        groups = df.groupby(col).median().reset_index()
        groups = groups.transpose()
        groups['diff'] = groups[1] - groups[0]

        norm = np.linalg.norm(groups['diff'].values)

        # Quantifying the repartition of data between df[df[col]==0] and df[df[col]==1]
        # This is measured via the entropy
        p0 = df[col].value_counts()[0] / Ltot
        p1 = df[col].value_counts()[1] / Ltot
        entropy = -p0 * math.log10(p0) - p1 * math.log10(p1)

        # Quantifying the repartition of data between the 4 categories:
        # df[df[col]==0 & df[VARIABLE_CIBLE]==0], df[df[col]==1 & df[VARIABLE_CIBLE]==0],
        # df[df[col]==0 & df[VARIABLE_CIBLE]==1] and df[df[col]==1 & df[VARIABLE_CIBLE]==1].
        # This is done via the entropy
        groups_count = df.groupby((col, 'VARIABLE_CIBLE')).count().reset_index().iloc[:, 3].values
        groups_count = np.asarray(groups_count)

        groups_proba = - groups_count / Ltot * np.log10(groups_count / Ltot)
        tot_entropy = sum(groups_proba)

        # computing the score:
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

#--------------------------------------------------------------------------------------------------------------


class MultiLayerNN():
    """
    Class to build and train multilayer perceptron with TensorFlow
    1-fold Cross Validation is performed while training, with the computation
    of a score, to determine when to stop the training.
    """

    def __init__(self, architect=None, dropouts=None):
        """
        inputs :

        architect = [n_input, n_h1, ..., n_hn, n_output]
        dropouts  = [True/False for (n_h1), ..., True/False for (n_hn)] : no dropouts on input or output layer

        """
        self.arch = architect
        self.drops = dropouts

        self.x = tf.placeholder("float", [None, self.arch[0]], name='x-input')
        self.y = tf.placeholder("float", [None, self.arch[-1]], name='y-input')

        self.nl = len(self.arch) - 1  # number of hidden layers + output layer
        self.weights = [tf.Variable(tf.random_normal([self.arch[i], self.arch[i + 1]]))
                        for i in range(self.nl)]
        self.biases = [tf.Variable(tf.random_normal([self.arch[i + 1]])) for i in range(self.nl)]

        self.keep_prob = tf.placeholder("float", name='keep-prob')
        self.roc_auc = tf.placeholder('float', name='roc_auc')

        # building the network
        self.pred = self.multilayer_perceptron(self.x, self.weights, self.biases, self.keep_prob)

        self.saver = tf.train.Saver()  # defaults to saving all variables

    def multilayer_perceptron(self, _X, _weights, _biases, _keep_prob):
        """
        function building the network
        """
        # first hidden layer
        layers = [tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights[0]), _biases[0]))]

        for i in range(self.nl - 1):
            if self.drops[i]:
                layers[-1] = tf.nn.dropout(layers[-1], _keep_prob)

            layers.append(tf.nn.sigmoid(
                tf.add(tf.matmul(layers[-1], _weights[i + 1]), _biases[i + 1])))

        print('number of hidden layers:', len(layers) - 1)
        return layers[-1]

    def set_cost(self, cost_function='squared_error'):

        if cost_function == 'squared_error':
            self.diff = tf.sub(self.pred, self.y)
            self.cost = tf.cast(tf.matmul(self.diff, self.diff, transpose_a=True), 'float')

    def set_optimizer(self, learning_rate):
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def set_summaries(self, sess, path):
        """
        This is the tracking of some quantities,
        to vizualize them in tensorboard.
        """
        # Add summary ops to collect data
        #w1_hist = tf.histogram_summary("weights_1", self.weights['h1'])
        #wout_hist = tf.histogram_summary("weights_out", weights['out'])
        #b1_hist = tf.histogram_summary("biases_1", biases['b1'])
        #bout_hist = tf.histogram_summary("biases_out", biases['out'])
        #y_hist = tf.histogram_summary("pred", pred)
        self.cost_summ = tf.histogram_summary("cost", self.cost)
        self.roc_summary = tf.histogram_summary("roc_auc", self.roc_auc)

        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(path, sess.graph_def)

    def fit(self, x_train, y_train, x_test, y_test, learning_rate, sess,
            training_epochs=10000, tol=1e-4,
            checkpoint_steps=5, display_step=1,
            keep_prob=0.5, cost_function='squared_error',
            path="/home/maxime/projects/challengeMDI343/sumWriter/",
            model_path='/home/maxime/projects/challengeMDI343/tensorflow_models/model.ckpt'):

        self.set_cost(cost_function=cost_function)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.set_summaries(sess, path=path)

        self.roc_aucs = [0., 0.1]
        self.costs = []

        init = tf.initialize_all_variables()
        sess.run(init)

        # Training cycle
        try:
            epoch = 0
            while epoch <= training_epochs and abs(self.roc_aucs[-1] - self.roc_aucs[-2]) >= tol:

                # Fit training using train data
                self.optimizer.run(
                    feed_dict={self.x: x_train, self.y: y_train, self.keep_prob: keep_prob})

                # Compute loss
                loss = self.cost.eval(
                    feed_dict={self.x: x_train, self.y: y_train, self.keep_prob: 1.0})

                # Roc_auc
                probs = self.pred.eval(feed_dict={self.x: x_test, self.keep_prob: 1.0})
                roc_au = roc_auc_score(y_test, probs[:, 0].reshape((-1, 1)))

                self.roc_aucs.append(roc_au)
                self.costs.append(loss)

                if epoch % checkpoint_steps == 0:
                    self.saver.save(sess, model_path, global_step=epoch)

                    # Record summary data, and the accuracy
                    feed = {self.x: x_train, self.y: y_train,
                            self.roc_auc: roc_au, self.keep_prob: 1.0}
                    result = self.merged.eval(feed_dict=feed)
                    self.writer.add_summary(result, epoch)

                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=",
                          "%7.3f" % (loss), "roc_auc:", roc_au)

                epoch += 1

        except KeyboardInterrupt:
            pass

    def predict(self, x_test):
        probs = self.pred.eval(feed_dict={self.x: x_test, self.keep_prob: 1.0})
        return probs

    def score(self, x_test, y_test):
        probs = self.pred.eval(feed_dict={self.x: x_test, self.keep_prob: 1.0})
        roc_au = roc_auc_score(y_test, probs[:, 0].reshape((-1, 1)))
        return roc_au
