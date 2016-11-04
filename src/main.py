import sys, os, nltk
import mailbox

from os import listdir
from scrape_mbox import scrape, path_to_takeout1, path_to_takeout2

def popular_ngrams(candidate_dict):
   top_trigrams = []
   top_fourgrams = []
   top_fivegrams = []
   for key,value in candidate_dict.items():
      if candidate_dict[key]['word_tokens'] is not None:
         trigrams = nltk.trigrams(candidate_dict[key]['word_tokens'])
         fdist = nltk.FreqDist(trigrams)
         print(fdist.most_common(5))

         fourgrams = nltk.ngrams(candidate_dict[key]['word_tokens'], 4)
         fdist = nltk.FreqDist(fourgrams)
         print(fdist.most_common(5))

         fivegrams = nltk.ngrams(candidate_dict[key]['word_tokens'], 5)
         fdist = nltk.FreqDist(fivegrams)
         print(fdist.most_common(5))
      else:
         print("Email had no payload")

if __name__ == '__main__':
   takeout1_mboxes = [path_to_takeout1 + "/" + f for f in listdir(path_to_takeout1)]
   takeout2_mboxes = [path_to_takeout2 + "/" + f for f in listdir(path_to_takeout2)]

   for mbox in takeout1_mboxes:
      print(mbox)
      candidate_dict = scrape(mbox)
      popular_ngrams(candidate_dict)
      break

   #for m in takeout2_mboxes:
   #   print(m)
