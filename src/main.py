import nltk
import mailbox
from os import listdir
from os.path import join
from scrape_mbox import scrape, path_to_takeout1, path_to_takeout2


def popular_ngrams(candidate_dict, cand=None):
    if not cand:
        cand = 'Unknown'
    trigrams = []
    fourgrams = []
    fivegrams = []
    for date, email in candidate_dict.items():
        if email['word_tokens']:
            trigrams.extend(nltk.trigrams(email['word_tokens']))
            fourgrams.extend(nltk.ngrams(email['word_tokens'], 4))
            fivegrams.extend(nltk.ngrams(email['word_tokens'], 5))

        else:
            print("Email had no payload")

    print('Candidate: {}'.format(cand))
    print('trigrams: {}'.format(', '.join(map(str, map(lambda x: x[0], nltk.FreqDist(trigrams).most_common(5))))))


if __name__ == '__main__':
    takeout1_mboxes = [(join(path_to_takeout1, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout1)]
    takeout2_mboxes = [(join(path_to_takeout2, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout2)]

    for mbox, cand in takeout1_mboxes[:1]:
        candidate_dict = scrape(mbox)
        popular_ngrams(candidate_dict, cand)

        # for m in takeout2_mboxes:
        #   print(m)
