import warnings
import copy
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
    all_words = test_set.wordlist
    Xlengths = copy.deepcopy(test_set.get_all_Xlengths())
    old_keys = list(Xlengths.keys())
    new_xlengths = {word: None for word in all_words}
    # Changing the keys of Xlengths as it's integers and not the words of the test set
    for i in range(len(all_words)):
        new_xlengths[all_words[i]] = Xlengths.pop(old_keys[i])

    for key, model in models.items():
        # Initializing an empty dictionary with word keys
        prob_dict = {word_key: None for word_key in all_words}
        # Score all words
        for word in all_words:
            # print(word)        
            try:
                logL = model.score(new_xlengths[word][0], new_xlengths[word][1])
            except:
                logL = float("-Inf")
            prob_dict[word] = logL
        # Find the best guess word
        best_guess = max(prob_dict)
        # Update probabilities and guesses list
        probabilities.append(prob_dict)
        guesses.append(best_guess)

    return probabilities, guesses
