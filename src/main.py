import nltk, mailbox
from nltk.corpus import stopwords as sw
from os import listdir
from os.path import join
from scrape_mbox import scrape, path_to_takeout1, path_to_takeout2

def extract_features1(candidate_tuple):
    print(candidate_tuple[1])
    words = []
    for date, email in candidate_tuple[0].items():
        if candidate_tuple[0][date]['word_tokens'] is not None:
            for word in candidate_tuple[0][date]['word_tokens']:
                words.append(word.lower())

    features = {}
    # lex_div gives accuracy of .416
    lex_div = len(set(words))/len(words)
    # print("len of set: " + str(len(set(words))) + " len words: " + str(len(words)) + " lex: " + str(lex_div))

    if lex_div <= 0.05:
        features['lex_div'] = 1
    elif lex_div > 0.05 and lex_div <= 0.065:
        features['lex_div'] = 2
    elif lex_div > 0.065 and lex_div <= 0.08:
        features['lex_div'] = 3
    elif lex_div > 0.08 and lex_div <= 0.095:
        features['lex_div'] = 4
    elif lex_div > 0.095 and lex_div <= 0.11:
        features['lex_div'] = 5
    elif lex_div > 0.11 and lex_div <= 0.125:
        features['lex_div'] = 6
    elif lex_div > 0.125 and lex_div <= 0.140:
        features['lex_div'] = 7
    elif lex_div > 0.140 and lex_div <= 0.155:
        features['lex_div'] = 8
    else:
        features['lex_div'] = 9

    trigrams, fourgrams, fivegrams = popular_ngrams(candidate_tuple[0], candidate_tuple[1])
    trigrams = [(trigram[0].lower(), trigram[1].lower(), trigram[2].lower()) for trigram in trigrams]
    fourgrams = [(fourgram[0].lower(), fourgram[1].lower(), fourgram[2].lower(), fourgram[3].lower()) for fourgram in fourgrams]
    for fourgram, count in nltk.FreqDist(fourgrams).most_common()[10:20]:
        features[fourgram] = True

    return features

def exercise1(train, test):
    train = [(extract_features1(candidate_tuple), candidate_tuple[1]) for candidate_tuple in train]
    print()
    test = [(extract_features1(candidate_tuple), candidate_tuple[1]) for candidate_tuple in test]

    classifier = nltk.NaiveBayesClassifier.train(train)
    print("Accuracy: " + str(nltk.classify.accuracy(classifier, test)))

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

        #else:
        #    print("Email had no payload")

    # print('Candidate: {}'.format(cand))
    # print('trigrams: {}'.format(', '.join(map(str, map(lambda x: x[0], nltk.FreqDist(trigrams).most_common(5))))))
    # print('fourgrams: {}'.format(', '.join(map(str, map(lambda x: x[0], nltk.FreqDist(fourgrams).most_common(5))))))
    # print('fivegrams: {}'.format(', '.join(map(str, map(lambda x: x[0], nltk.FreqDist(fivegrams).most_common(5))))))
    
    return trigrams, fourgrams, fivegrams

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
    sents = list(set(sents))
    print(sents[:10])

if __name__ == '__main__':
    takeout1_mboxes = [(join(path_to_takeout1, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout1)]
    takeout2_mboxes = [(join(path_to_takeout2, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout2)]

    train = []
    test = []

    for mbox, cand in takeout1_mboxes:
        candidate_tuple = (scrape(mbox), cand)
        train.append(candidate_tuple)

        # 1
        #popular_ngrams(candidate_dict, cand)
        # 2
        #money_talk(candidate_dict, cand) 

    for mbox, cand in takeout2_mboxes:
        candidate_tuple = (scrape(mbox), cand)
        test.append(candidate_tuple)

    exercise1(train,test)    
