import email
import glob
import numpy as np
from operator import is_not
from functools import partial
from sklearn.model_selection import train_test_split
def test(vectorizer):
    # Load the path for each email file for both categories.
    ham_files = train_test_split(glob.glob('./data/20030228_hard_ham/hard_ham/*'), random_state=123)
    spam_files = train_test_split(glob.glob('./data/20050311_spam_2/spam_2/*'), random_state=123)

    # Method for getting the content of an email.
    def get_content(filepath):
        file = open(filepath, encoding='latin1')
        message = email.message_from_file(file)
        
        for msg_part in message.walk():
            # Keep only messages with text/plain content.
            if msg_part.get_content_type() == 'text/plain':
                return msg_part.get_payload()

    # Get the training and testing data.
    ham_train_data = [get_content(i) for i in ham_files[0]]
    ham_test_data = [get_content(i) for i in ham_files[1]]
    spam_train_data = [get_content(i) for i in spam_files[0]]
    spam_test_data = [get_content(i) for i in spam_files[1]]

    # Keep emails with non-empty content.
    ham_train_data = list(filter(None, ham_train_data))
    ham_test_data = list(filter(None, ham_test_data))
    spam_train_data = list(filter(None, spam_train_data))
    spam_test_data = list(filter(None, spam_test_data))

    # Merge the train/test files for both categories.
    train_data = np.concatenate((ham_train_data, spam_train_data))
    test_data = np.concatenate((ham_test_data, spam_test_data))

    # Assign a class for each email (ham = 0, spam = 1).
    ham_train_class = [0]*len(ham_train_data)
    ham_test_class = [0]*len(ham_test_data)
    spam_train_class = [1]*len(spam_train_data)
    spam_test_class = [1]*len(spam_test_data)

    # Merge the train/test classes for both categories.
    train_class = np.concatenate((ham_train_class, spam_train_class))
    test_class = np.concatenate((ham_test_class, spam_test_class))

    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # Tokenize the train/test data.
    train_data = [word_tokenize(i) for i in train_data]
    test_data = [word_tokenize(i) for i in test_data]


    # Method for removing the stop words.
    def remove_stop_words(input):
        result = [i for i in input if i not in ENGLISH_STOP_WORDS]
        return result

    # Remove the stop words.
    train_data = [remove_stop_words(i) for i in train_data]
    test_data = [remove_stop_words(i) for i in test_data]

    # Create the lemmatizer.
    lemmatizer = WordNetLemmatizer()

    # Method for lemmatizing the text.
    def lemmatize_text(input):
        return [lemmatizer.lemmatize(i) for i in input]

    # Lemizuj tekst.
    train_data = [lemmatize_text(i) for i in train_data]
    test_data = [lemmatize_text(i) for i in test_data]

    # Zrekonstruj dane.
    train_data = [" ".join(i) for i in train_data]
    test_data = [" ".join(i) for i in test_data]


    # Transform the test/train data into features.
    train_data_features = vectorizer.transform(train_data)
    test_data_features = vectorizer.transform(test_data)
    return test_data_features,test_class