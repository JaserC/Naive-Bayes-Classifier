import numpy as np
from typing import List, Set, Dict
import os

class NaiveBayes():
 
    def __init__(self):
        self.num_train_hams = 0
        self.num_train_spams = 0
        self.word_counts_spam = {}
        self.word_counts_ham = {}
        self.HAM_LABEL = 'ham'
        self.SPAM_LABEL = 'spam'

    def load_data(self, path:str='data/'):
        assert set(os.listdir(path)) == set(['test', 'train'])
        assert set(os.listdir(os.path.join(path, 'test'))) == set(['ham', 'spam'])
        assert set(os.listdir(os.path.join(path, 'train'))) == set(['ham', 'spam'])

        train_hams, train_spams, test_hams, test_spams = [], [], [], []
        for filename in os.listdir(os.path.join(path, 'train', 'ham')):
            train_hams.append(os.path.join(path, 'train', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'train', 'spam')):
            train_spams.append(os.path.join(path, 'train', 'spam', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'ham')):
            test_hams.append(os.path.join(path, 'test', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'spam')):
            test_spams.append(os.path.join(path, 'test', 'spam', filename))

        return train_hams, train_spams, test_hams, test_spams

    def word_set(self, filename:str) -> Set[str]:
        with open(filename, 'r') as f:
            text = f.read()[9:] # Ignoring 'Subject:'
            text = text.replace('\r', '')
            text = text.replace('\n', ' ')
            words = text.split(' ')
            return set(words)

    def fit(self, train_hams:List[str], train_spams:List[str]):
        self.num_train_hams = len(train_hams)
        self.num_train_spams = len(train_spams)
        def get_counts(filenames:List[str]) -> Dict[str, int]:
            word_count = {}
            for file in filenames:
                words = self.word_set(file)
                for word in words: 
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1

            return word_count
        
        self.word_counts_ham = get_counts(train_hams)
        self.word_counts_spam = get_counts(train_spams)

    def predict(self, filename:str) -> str:
        def calculate_prob(word_counts:Dict[str, int], num_total:int, filename:str) -> float:
            prob = 0
            words = self.word_set(filename)
            for word in words:
                count = word_counts.get(word, 0) + 1
                prob += np.log(count / (num_total + 2))  # Laplace smoothing
            return prob

        total = self.num_train_hams + self.num_train_spams

        ham_prob = np.log(self.num_train_hams / total) + calculate_prob(self.word_counts_ham, self.num_train_hams, filename)
        spam_prob = np.log(self.num_train_spams / total) + calculate_prob(self.word_counts_spam, self.num_train_spams, filename)

        return self.HAM_LABEL if ham_prob >= spam_prob else self.SPAM_LABEL
    
    def accuracy(self, hams:List[str], spams:List[str]) -> float:
        total_correct = 0
        total_datapoints = len(hams) + len(spams)
        for filename in hams:
            if self.predict(filename) == self.HAM_LABEL:
                total_correct += 1
        for filename in spams:
            if self.predict(filename) == self.SPAM_LABEL:
                total_correct += 1
        return total_correct / total_datapoints

if __name__ == '__main__':
    nbc = NaiveBayes()

    train_hams, train_spams, test_hams, test_spams = nbc.load_data()

    nbc.fit(train_hams, train_spams)
    
    print("Training Accuracy: {}".format(nbc.accuracy(train_hams, train_spams)))
    print("Test  Accuracy: {}".format(nbc.accuracy(test_hams, test_spams)))
