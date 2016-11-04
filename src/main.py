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
    # print('fourgrams: {}'.format(', '.join(map(str, map(lambda x: x[0], nltk.FreqDist(fourgrams).most_common(5))))))
    # print('fivegrams: {}'.format(', '.join(map(str, map(lambda x: x[0], nltk.FreqDist(fivegrams).most_common(5))))))

def money_talk(candidate_dict, cand=None):
    if not cand:
        cand = 'Unknown'
    sents = []
    for date, email in candidate_dict.items():
        if email['sent_tokens']:
            for sent in email['sent_tokens']:
                if '$' in sent or 'money' in sent or 'donate' in sent:
                    sents.append(sent)

    print('Candidate: {}'.format(cand))
    print(sents[:2])

if __name__ == '__main__':
    takeout1_mboxes = [(join(path_to_takeout1, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout1)]
    takeout2_mboxes = [(join(path_to_takeout2, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout2)]

    for mbox, cand in takeout1_mboxes:
        candidate_dict = scrape(mbox)
        # Interesting thing 1
        popular_ngrams(candidate_dict, cand)

        # Interesting thing 2
        money_talk(candidate_dict, cand) 

        # Interesting thing 3

    # for m in takeout2_mboxes:
    #   print(m)
