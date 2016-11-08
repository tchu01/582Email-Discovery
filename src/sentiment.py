import os
import re
import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from scrape_mbox import scrape
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


_data_dir = '../data/Takeout 3/Mail'


def show(data_dir=_data_dir):
    mailboxes = [os.path.join(_data_dir, d) for d in os.listdir(_data_dir)]
    cand_ids = set(mb.split('/')[-1].split('.')[0].split('-')[-1].lower() for mb in mailboxes)
    cand_data = map(scrape, mailboxes)
    cand_dict = defaultdict(list)
    sa = SentimentIntensityAnalyzer()

    for n, data in enumerate(cand_data):
        cand = mailboxes[n].split('/')[-1].split('.')[0].split('-')[-1].lower()
        for date, email in data.items():
            if email:
                if email['sent_tokens']:
                    X_ = [0 for _ in range(len(cand_ids))]
                    for sent in [x.lower() for x in email['sent_tokens']]:
                        for n_, cand_ in enumerate(cand_ids):
                            if cand_ in sent:
                                X_[n_] += sa.polarity_scores(sent)['compound']
                    cand_dict[cand].append(X_)

    data_ = []
    index = []
    a = -1
    b = 1

    for cand_id, data in cand_dict.items():
        d = np.asarray(data).sum(axis=0)
        d_ = ((b - a) * (d - d.min())) / (d.max() - d.min())
        d_ += a
        d_[np.where(d == 0)] = 0
        data_.append(d_.tolist())
        index.append(cand_id)

    df = pd.DataFrame(data_, index=index, columns=cand_ids)
    ax = df.plot(colormap='Set1', figsize=(16, 10), title='Candidate Sentiments')

    ax.set_xlabel('candidate')
    ax.set_ylabel('sentiment')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
    plt.savefig('../results/sentiment.png')
    plt.close()

    X = []
    y = []

    for cand_id, data in cand_dict.items():
        d_ = np.asarray(data)
        idcs = np.random.choice(range(d_.shape[0]), 10)
        X.extend(d_[idcs].tolist())
        y.extend([cand_id for _ in range(10)])

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

    with open('../results/sentiment.txt', 'w') as f:
        f.write(classification_report(y_true, y_predicted))
        f.write('Accuracy: {:.4f}'.format(sum([1 if x == y else 0 for x, y in zip(y_true, y_predicted)]) / len(y_true)))


if __name__ == '__main__':
    show()