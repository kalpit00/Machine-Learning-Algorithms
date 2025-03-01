from typing import Tuple

import numpy as np
import pandas as pd


def load_data(path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the data from dating_train.csv. 
    Preprocess the data by performing the following:
    (i) Remove the quotes present in columns "race","race_o","field"
    (ii) Convert all the values in the column "field" to lowercase
    (iii) Perform Label Encoding for the columns 'gender','race','race_o','field'.
    (iv) Perform normalization. For each row, let's first sum up all the values in the 
      six columns that belong to the set preference scores of participant (denote the sum value as total),
      and then transform the value for each column in the set preference scores of participant in that row as follows: new value=old value/total.
      We then conduct similar transformation for values in the set preference scores of partner.


    Args:
        path: path to the dataset

    Returns:
        X: data
        y: labels
    """

    dataset = pd.read_csv(path)
    df1 = dataset.copy()

    # Removal of quotes
    def _remove_quotes():
        columns = ["race", "race_o", "field"]
        count = 0
        # >>> YOUR CODE HERE >>>
        for j in columns:
            for i in range(len(dataset[j])):
                if "'" in list(dataset[j].values[i]):
                    dataset[j].values[i] = dataset[j].values[i].strip("'")
                    dataset[j].values[i] = dataset[j].values[i].strip(" ")
                    count = count+1
        # <<< END OF YOUR CODE <<<
        print('Quotes removed from ' + str(count) + ' cells')

    # Convert all the values in the column field to lowercase

    def _to_lower_case():
        count = 0
        # >>> YOUR CODE HERE >>>
        for i in range(len(dataset['field'])):
            if dataset['field'].values[i].islower() == False:
                dataset['field'].values[i] = dataset['field'].values[i].lower()
                count = count+1
        # <<< END OF YOUR CODE <<<
        print('Standardized ' + str(count) + ' cells to lower case')

    # Label Encoding

    def _label_encoding():
        cols = ['gender', 'race', 'race_o', 'field']
        # >>> YOUR CODE HERE >>>
        for i in cols:
            dataset[i] = dataset[i].astype('category')
            dataset[i] = dataset[i].cat.codes

            for i in range(len(dataset)):
                if (df1['gender'].values[i] == 'male'):
                    ind1 = i
                if (df1['race'].values[i] == 'European/Caucasian-American'):
                    ind2 = i
                if (df1['race_o'].values[i] == '\'Latino/Hispanic American\''):
                    ind3 = i
                if (df1['field'].values[i] == 'Law'):
                    ind4 = i
        # <<< END OF YOUR CODE <<<
        print("Value assigned for male in column gender:",
              dataset['gender'].values[ind1])
        print("Value assigned for European/Caucasian-American in column race:",
              dataset['race'].values[ind2])
        print("Value assigned for Latino/Hispanic American in column race_o:",
              dataset['race_o'].values[ind3])
        print("Value assigned for law in column field:",
              dataset['field'].values[ind4])

    # Normalization
    def _normalize():
        preference_partner_parameters = ['pref_o_attractive', 'pref_o_sincere',
                                         'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
        preference_participant_parameters = ['attractive_important', 'sincere_important',
                                             'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
        sum_preference_partner_scores = []
        sum_preference_participant_scores = []
        preference_scores_of_participant_mean = []
        preference_scores_of_partner_mean = []

        # Calculating mean for preference_scores_of_partner
        for j in preference_partner_parameters:
            for i in range(len(dataset)):
                # >>> YOUR CODE HERE >>>
                sum_preference_partner_scores.append(dataset['pref_o_attractive'].values[i]+dataset['pref_o_sincere'].values[i]+dataset['pref_o_intelligence'].values[i] +
                                                     dataset['pref_o_ambitious'].values[i]+dataset['pref_o_funny'].values[i]+dataset['pref_o_shared_interests'].values[i])
                dataset[j].values[i] = dataset[j].values[i] / \
                    sum_preference_partner_scores[i]
                # <<< END OF YOUR CODE <<<

        # Calculating mean for preference_scores_of_participant
        for k in preference_participant_parameters:
            for i in range(len(dataset)):
                # >>> YOUR CODE HERE >>>
                sum_preference_participant_scores.append(dataset['attractive_important'].values[i]+dataset['sincere_important'].values[i]+dataset['intelligence_important'].values[i] +
                                                         dataset['funny_important'].values[i]+dataset['ambition_important'].values[i]+dataset['shared_interests_important'].values[i])
                dataset[k].values[i] = dataset[k].values[i] / \
                    sum_preference_participant_scores[i]
                # <<< END OF YOUR CODE <<<

        # Calculating the mean values of each columns in  preference_scores_of_partner
        for x in preference_partner_parameters:
            preference_scores_of_partner_mean.append(dataset[x].mean())

            # Calculating the mean values of each columns in  preference_scores_of_participant
        for y in preference_participant_parameters:
            preference_scores_of_participant_mean.append(dataset[y].mean())

            # Print the values of mean of each column
        print("Mean of attractive_important:" +
              str(round(preference_scores_of_participant_mean[0], 2)))
        print("Mean of sincere_important:" +
              str(round(preference_scores_of_participant_mean[1], 2)))
        print("Mean of intelligence_important:" +
              str(round(preference_scores_of_participant_mean[2], 2)))
        print("Mean of funny_important:" +
              str(round(preference_scores_of_participant_mean[3], 2)))
        print("Mean of ambition_important:" +
              str(round(preference_scores_of_participant_mean[4], 2)))
        print("Mean of shared_interests_important:" +
              str(round(preference_scores_of_participant_mean[5], 2)))

        print('Mean of pref_o_attractive:' +
              str(round(preference_scores_of_partner_mean[0], 2)))
        print('Mean of pref_o_sincere:' +
              str(round(preference_scores_of_partner_mean[1], 2)))
        print('Mean of pref_o_intelligence:' +
              str(round(preference_scores_of_partner_mean[2], 2)))
        print('Mean of pref_o_funny:' +
              str(round(preference_scores_of_partner_mean[3], 2)))
        print('Mean of pref_o_ambition:' +
              str(round(preference_scores_of_partner_mean[4], 2)))
        print('Mean of pref_o_shared_interests:' +
              str(round(preference_scores_of_partner_mean[5], 2)))

    _remove_quotes()
    _to_lower_case()
    _label_encoding()
    _normalize()

    X = dataset.drop(columns=['decision']).values
    y = dataset['decision'].values

    return X, y


def shuffle_data(X, y, random_state=None):
    """
    Shuffle the data.

    Args:
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )
        seed: int or None

    Returns:
        X: shuffled data
        y: shuffled labels
    """
    if random_state:
        np.random.seed(random_state)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]
