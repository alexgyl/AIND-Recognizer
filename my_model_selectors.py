import math
import statistics
import warnings
import copy # Used for deep copy

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ## Implement model selection using BIC
        # Initial values
        best_score = float("Inf")
        best_num_states = 2
        ## Iterate through a number of states to test which is the best representation
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            BIC_score = 0
            
            try:
                # Catch case if n_samples > n_states
                # if len(self.X) < num_states:
                if num_states > sum(self.lengths):
                    return None
                else:
                    # print(self.this_word)
                    # print("Number of samples {}".format(sum(self.lengths)))
                    # print("Length of self.x[0] {}".format(len(self.X[0])))
                    # print("Length of self.x {}".format(len(self.X)))
                    # print("Shape size X[0] {}".format(self.X.shape[0]))
                    # print("Number of states {}".format(num_states))
                    # HMM Model building - num_states is our parameter that is found using CV
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, num_states))
                    # Log-likelihood score
                    logL = hmm_model.score(self.X, self.lengths)
                    # Number of parameters used by the model - HMMs are defined by the transition probabilities,
                    # the emission probabilities, initial probability, means and variance of distribution
                    # Let n be the number of states and m be the number of features
                    # Transition probabilities -> n * (n - 1) since for the last prob, we can find it through (1 - all other prob)
                    # Initial probabilites -> n - 1 since we have n possible states to start in but last state can found via (1 - n)
                    # Means of distributions -> n * m means as there is a distribution for each features in each state
                    # Variance of distributions -> n * m variances as there needs to be a variance for each distribution and we are 
                    # also using normal distributions
                    # This gives us n^2 + 2nm - 1
                    n_samples, n_features = self.X.shape
                    n_params = num_states ** 2 + (2 * num_states * n_features) - 1

                    # BIC score
                    BIC_score = (-2 * logL) + (n_params * math.log(n_samples))
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                    return None
            ## Tracking best score and best number of states parameter - the lower the BIC the better
            if BIC_score < best_score:
                best_score = BIC_score
                best_num_states = num_states

        ## Build the best hmm model using all data once parameter has been finalized
        # print("CURRENT WORD: {}".format(self.this_word))
        best_hmm_model = GaussianHMM(n_components=best_num_states, covariance_type="diag", n_iter=1000,
                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        if self.verbose:
            print("Best model created for {} with {} states".format(self.this_word, best_num_states))

        return best_hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores
        # Initial values
        best_score = float("-Inf")
        best_num_states = 2
        ## Iterate through a number of states to test which is the best representation
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            DIC_score = 0

            try:
                # HMM Model building - num_states is our parameter that is found using DIC
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))                
                # Log-likelihood of model in context of evidence
                logL = hmm_model.score(self.X, self.lengths)
                # Remove current word from the list of all words 
                anti_words = list(self.words.keys())
                anti_words.remove(self.this_word)
                # Log-likelihood of model in context of anti-evidence
                anti_scores = sum([hmm_model.score(self.hwords[word][0], self.hwords[word][1]) for word in anti_words])
                # DIC Score
                DIC_score = logL - (anti_scores / len(anti_words))
            except:   
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                    return None

            ## Tracking best score and best number of states parameter - the higher the DIC the better
            if DIC_score > best_score:
                best_score = DIC_score
                best_num_states = num_states

        # Build the best hmm model using all data once parameter has been finalized
        best_hmm_model = GaussianHMM(n_components=best_num_states, covariance_type="diag", n_iter=1000,
                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        if self.verbose:
            print("Best model created for {} with {} states".format(self.this_word, best_num_states))

        return best_hmm_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        ## Implement model selection using CV
        # Initial values
        best_score = float("-Inf")
        best_num_states = 2
        ## Iterate through a number of states to test which is the best representation
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            score = 0
            ## Perform CV split
            # Case if we have less data than samples (here we use KFold = 3 folds)
            if len(self.sequences) < 3:
                n_split = 1
                DO_KFOLD = False
            else:
                n_split = 3
                DO_KFOLD = True
            ## Decide if need to do K-Fold as per the data       
            if DO_KFOLD:
            ## Create the split method & fix the seeding
                split_method = KFold(n_splits = n_split, random_state = self.random_state)
                ## KFold CV
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        # Split the sequences accordingly using helper function to split the data correct
                        sequence_split_train = combine_sequences(cv_train_idx, self.sequences)
                        sequence_split_cv = combine_sequences(cv_test_idx, self.sequences)
                        # HMM Model building - num_states is our parameter that is found using CV
                        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(sequence_split_train[0], sequence_split_train[1])
                        if self.verbose:
                            print("model created for {} with {} states".format(self.this_word, num_states))
                        score += hmm_model.score(sequence_split_cv[0], sequence_split_cv[1])
                    except:
                        if self.verbose:
                            print("failure on {} with {} states".format(self.this_word, num_states))
                            return None
            else:
                try:
                    # HMM Model building without CV
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, num_states))
                    score += hmm_model.score(self.X, self.lengths)
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
                        return None
            ## Average the score across all the folds - KFold is fixed to 3 at the moment, except if there's too little data
            score /= n_split
            ## Tracking best score and best number of states parameter
            if score > best_score:
                best_score = score
                best_num_states = num_states

        ## Build the best hmm model using all data once parameter has been finalized
        best_hmm_model = GaussianHMM(n_components=best_num_states, covariance_type="diag", n_iter=1000,
                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        if self.verbose:
            print("Best model created for {} with {} states".format(self.this_word, best_num_states))

        return best_hmm_model
