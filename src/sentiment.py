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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


_data_dirs = ['../data/Takeout 3/Mail']


def show(data_dirs=_data_dirs):
    mailboxes = [x for y in
                 [[os.path.join(data_dir, d) for d in os.listdir(data_dir)] for data_dir in data_dirs]
                 for x in y]
    cand_ids = list(set(mb.split('/')[-1].split('.')[0].split('-')[-1].lower() for mb in mailboxes))
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
                            if cand_ in sent and cand_ != cand:
                                X_[n_] += sa.polarity_scores(sent)['compound']
                    cand_dict[cand].append(X_)

    data_ = []
    a = -1
    b = 1
    for data in [cand_dict[cid] for cid in cand_ids]:
        d = np.asarray(data).sum(axis=0)
        d_ = ((b - a) * (d - d.min())) / (d.max() - d.min())
        d_ += a
        d_[np.where(d == 0)] = 0
        data_.append(d_.tolist())

    # this makes a flat line for each candidates average sentiment. we might want to plot this
    values = np.asarray(data_).mean(axis=0)
    values = (((5 + 5) * (values - values.min())) / (values.max() - values.min())) - 5
    x_ = sorted(range(values.size), key=lambda n: values[n])
    
    # plot each candidates sentiment line
    df = pd.DataFrame(data_, columns=cand_ids, index=cand_ids)
    ax = df.plot(colormap='Set1', figsize=(16, 10), title='Candidate Sentiments')
    ax.set_xlabel('candidate')
    ax.set_ylabel('sentiment')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)
    plt.savefig('../results/sentiment.png')
    plt.clf()

    X = []
    y = []

    # exclude candidates that haven't expressed 100 sentiments
    keep_idcs = [n for n in range(len(cand_ids)) if len(cand_dict[cand_ids[n]]) >= 100]
    cand_dict = {k: v for k, v in cand_dict.items() if len(v) >= 100}
    for cand_id, data in cand_dict.items():
        d_ = np.asarray(data)[:, keep_idcs]
        idcs = np.random.choice(range(d_.shape[0]), 100)
        X.extend(d_[idcs].tolist())
        y.extend([cand_id for _ in range(100)])

    X = np.asarray(X)
    y = np.asarray(y)

    pipe = Pipeline([('scale', StandardScaler()), ('clf', LinearSVC())])

    y_true = []
    y_predicted = []

    kfold = StratifiedKFold(n_splits=10)

    for train, test in kfold.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        pipe.fit(X_train, y_train)

        y_true.extend(y_test)
        y_predicted.extend(pipe.predict(X_test))

    with open('../results/sentiment.txt', 'w') as f:
        f.write(classification_report(y_true, y_predicted))


if __name__ == '__main__':
    show()