import os
import re
import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from scrape_mbox import scrape
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


_data_dir = '../data/'


def _walk(dir_path):
    result = []

    for dirpath, dirnames, filenames in os.walk(dir_path):
        for dn in dirnames:
            _walk(os.path.join(dirpath, dn))

        result.extend([os.path.join(dirpath, fn) for fn in filenames if re.match('(Republicans|Democrats).*', fn)])

    return result


def show(data_dir=_data_dir):
    mailboxes = _walk(data_dir)
    cand_ids = set(mb.split('/')[-1].split('.')[0].split('-')[-1].lower() for mb in mailboxes)
    cand_data = map(scrape, mailboxes)
    cand_dict = defaultdict(list)

    for n, data in enumerate(cand_data):
        cand = mailboxes[n].split('/')[-1].split('.')[0].split('-')[-1].lower()
        for date, email in data.items():
            if email:
                if email['word_tokens']:
                    X_ = [0 for _ in range(len(cand_ids))]
                    fd = nltk.FreqDist([x.lower() for x in email['word_tokens']])
                    for n_, cand_ in enumerate(cand_ids):
                        if cand_ != cand:
                            X_[n_] = float(fd[cand_])

                    cand_dict[cand].append(X_)

    data_ = []
    index = []

    for cand_id, data in cand_dict.items():
        d_ = np.asarray(data).sum(axis=0)
        d_ /= sum(d_)
        data_.append(d_.tolist())
        index.append(cand_id)

    df = pd.DataFrame(data_, index=index, columns=list(cand_ids))
    df.plot.bar(colormap='Set1')
    plt.xlabel('candidate')
    plt.ylabel('mention distribution')
    plt.show()
    plt.close()

    X = []
    y = []

    for cand_id, data in cand_dict.items():
        X.extend(data)
        y.extend([cand_id for _ in range(len(data))])

    X = np.asarray(X)
    y = np.asarray(y)

    pipe = Pipeline([('scale', StandardScaler()), ('clf', LinearSVC())])

    y_true = []
    y_predicted = []

    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        pipe.fit(X_train, y_train)

        y_true.extend(y_test)
        y_predicted.extend(pipe.predict(X_test))

    print(classification_report(y_true, y_predicted))

    print('Accuracy: {:.4f}'.format(sum([1 if x == y else 0 for x, y in zip(y_true, y_predicted)]) / len(y_true)))


if __name__ == '__main__':
    show()