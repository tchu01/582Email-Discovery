import nltk, mailbox, numpy
from nltk.corpus import stopwords as sw
from os import listdir
from os.path import join
from scrape_mbox import scrape, path_to_takeout1, path_to_takeout2, path_to_takeout3

def extract_features1(candidate_messages, cand):
    words = []
    for date, email_dict in candidate_messages:
        if email_dict['word_tokens'] is not None:
            for word in email_dict['word_tokens']:
                words.append(word.lower())

    features = {}
    '''
    # lex_div gives accuracy of .416
    lex_div = len(set(words))/len(words)
    print("len of set: " + str(len(set(words))) + " len words: " + str(len(words)) + " lex: " + str(lex_div))

    features['lex_div'] = lex_div
    '''

    trigrams, fourgrams, fivegrams = popular_ngrams(candidate_messages, cand)
    trigrams = [(trigram[0].lower(), trigram[1].lower(), trigram[2].lower()) for trigram in trigrams]
    fourgrams = [(fourgram[0].lower(), fourgram[1].lower(), fourgram[2].lower(), fourgram[3].lower()) for fourgram in fourgrams]
    for fourgram, count in nltk.FreqDist(fourgrams).most_common()[:10]:
        print(fourgram)
        features[fourgram] = True

    return features

def exercise1(train, test):
    processed_train = []
    processed_test = []

    for candidate_messages, cand in train:
        if len(candidate_messages) > 0:
            processed_train.append((extract_features1(candidate_messages, cand), cand))

    print()

    for candidate_messages, cand in test:
        if len(candidate_messages) > 0:
            processed_test.append((extract_features1(candidate_messages, cand), cand))

    '''
    train = [(extract_features1(candidate_tuple), candidate_tuple[1]) for candidate_tuple in train]
    print()
    test = [(extract_features1(candidate_tuple), candidate_tuple[1]) for candidate_tuple in test]
    '''

    classifier = nltk.NaiveBayesClassifier.train(processed_train)
    print("Accuracy: " + str(nltk.classify.accuracy(classifier, processed_test)))
    print(classifier.show_most_informative_features(20))

def popular_ngrams(candidate_messages, cand=None):
    if not cand:
        cand = 'Unknown'
    trigrams = []
    fourgrams = []
    fivegrams = []
    for date, email in candidate_messages:
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
    takeout3_mboxes = [(join(path_to_takeout3, f), f.split('.')[0].split('-')[-1]) for f in listdir(path_to_takeout3)]

    '''
    cands1 = []
    print("Takeout1")
    for mbox, cand in takeout1_mboxes:
        candidate_tuple = (scrape(mbox), cand)
        cands1.append(candidate_tuple)
        
        print("Candidate: " + candidate_tuple[1])
        print(len(candidate_tuple[0]))

        # 1
        #popular_ngrams(candidate_dict, cand)
        # 2
        #money_talk(candidate_dict, cand) 

    cands2 = []
    print()
    print("Takeout2")
    for mbox, cand in takeout2_mboxes:
        candidate_tuple = (scrape(mbox), cand)
        cands2.append(candidate_tuple)

        print("Candidate: " + candidate_tuple[1])
        print(len(candidate_tuple[0]))
    '''
    
    cands = []
    train = []
    test = []
    print("Takeout3")
    for mbox, cand in takeout3_mboxes:
        candidate_tuple = (scrape(mbox), cand)
        cands.append(candidate_tuple)

        print("Candidate: " + candidate_tuple[1])
        print(len(candidate_tuple[0]))

    for cand_dict, cand in cands:
        split = len(cand_dict) // 3
        items = list(cand_dict.items())
        
        train.append((items[split:], cand))
        test.append((items[:split], cand))
    
    #print(test[12])

    exercise1(train,test)    
