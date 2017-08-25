import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Get list of all test words and the sequences
    all_words = test_set.wordlist
    Xlengths = test_set.get_all_Xlengths()

    # Loop through all test set items
    for i in range(test_set.num_items):
        # Probability dictionary
        prob_dict = {}
        # Loop through all models and calculate the log-likelihod, if unable to, give worse case score
        for key, model in models.items():
            try: 
                logL = model.score(Xlengths[i][0], Xlengths[i][1])
            except:
                logL = float("-Inf")
            prob_dict[key] = logL
        # Get the word with the largest probability
        best_guess = max(prob_dict)
        # Update the lists accordingly
        probabilities.append(prob_dict)
        guesses.append(best_guess)

    return probabilities, guesses
