import email
import glob
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def test(vectorizer):
    # Wczytaj ścieżki do każdego pliku email dla obu kategorii.
    ham_files = glob.glob('./data/20030228_hard_ham/hard_ham/*')
    spam_files = glob.glob('./data/20050311_spam_2/spam_2/*')

    # Metoda do pobierania zawartości wiadomości e-mail.
    def get_content(filepath):
        file = open(filepath, encoding='latin1')
        message = email.message_from_file(file)
        
        for msg_part in message.walk():
            # Pomiń wiadomości, które nie posiadają zawartości tekstowej.
            if msg_part.get_content_type() == 'text/plain':
                return msg_part.get_payload()

    # Uzyskaj dane.
    ham_data = [get_content(i) for i in ham_files]
    spam_data = [get_content(i) for i in spam_files]

    # Zachowaj e-maile z niepustą treścią.
    ham_data = list(filter(None, ham_data))
    spam_data = list(filter(None, spam_data))

    # Połącz pliki obu kategorii.
    data = np.concatenate((ham_data, spam_data))

    # Przypisz klasę do każdego e-maila (ham = 0, spam = 1).
    ham_class = [0]*len(ham_data)
    spam_class = [1]*len(spam_data)

    # Połącz klasy dla obu kategorii.
    data_class = np.concatenate((ham_class, spam_class))

    # Tokenizuj dane.
    data = [word_tokenize(i) for i in data]

    # Metoda do usuwania stop words.
    def remove_stop_words(input):
        result = [i for i in input if i not in ENGLISH_STOP_WORDS]
        return result

    # Usuń stop words.
    data = [remove_stop_words(i) for i in data]

    # Metoda do lematyzacji tekstu.
    lemmatizer = WordNetLemmatizer()

    # Metoda do lematyzacji tekstu.
    def lemmatize_text(input):
        return [lemmatizer.lemmatize(i) for i in input]

    # Lemizuj tekst.
    data = [lemmatize_text(i) for i in data]

    # Zrekonstruj dane.
    data = [" ".join(i) for i in data]


    # Przekształć dane  w funkcje.
    data_features = vectorizer.transform(data)
    return data_features, data_class