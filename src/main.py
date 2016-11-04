import nltk
import mailbox
from os import listdir
from os.path import join
from scrape_mbox import scrape, path_to_takeout1, path_to_takeout2


def popular_ngrams(candidate_dict, cand=None):
    if not cand:
        cand = 'Unknown'
    top_trigrams = []
    top_fourgrams = []
    top_fivegrams = []
    for date, email in candidate_dict.items():
        if email['word_tokens']:
            trigrams = nltk.trigrams(email['word_tokens'])
            top_trigrams.extend(map(lambda x: x[0], nltk.FreqDist(trigrams).most_common(5)))

            fourgrams = nltk.ngrams(email['word_tokens'], 4)
            top_fourgrams.extend(map(lambda x: x[0], nltk.FreqDist(fourgrams).most_common(5)))

            fivegrams = nltk.ngrams(email['word_tokens'], 5)
            top_fivegrams.extend(map(lambda x: x[0], nltk.FreqDist(fivegrams).most_common(5)))

        else:
            print("Email had no payload")

    print(cand)
    print('trigrams: {}'.format(', '.join(map(str, top_trigrams))))


if __name__ == '__main__':
    takeout1_mboxes = [(join(path_to_takeout1, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout1)]
    takeout2_mboxes = [(join(path_to_takeout2, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout2)]

    for mbox, cand in takeout1_mboxes[:1]:
        candidate_dict = scrape(mbox)
        popular_ngrams(candidate_dict, cand)

        # for m in takeout2_mboxes:
        #   print(m)
