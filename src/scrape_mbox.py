import os, sys, nltk, re
import mailbox
from os import listdir
from collections import defaultdict

path_to_takeout1 = os.path.abspath("../data/Takeout1/Mail")
path_to_takeout2 = os.path.abspath("../data/Takeout2/Mail")

def scrape(mbox_filename):
    candidate_dict = {}
    mbox = mailbox.mbox(mbox_filename)

    for message in mbox:
        date = message['date']
        subject = message['subject']
        text_payload = clean_body(get_body(message))

        candidate_dict[date] = {}
        candidate_dict[date]['subject'] = subject
        candidate_dict[date]['payload'] = text_payload

        if text_payload:
            candidate_dict[date]['word_tokens'] = nltk.word_tokenize(text_payload)
            candidate_dict[date]['sent_tokens'] = nltk.sent_tokenize(text_payload)
        else:
            candidate_dict[date]['word_tokens'] = None
            candidate_dict[date]['sent_tokens'] = None

        # print("Date: " + str(date))
        # print("Subject: " + str(subject))
        # print("Payload: " + str(candidate_dict[date]['without_links']))
        # print("Tokens: " + str(nltk.word_tokenize(text_payload)))
        # print("Bigrams: " + str(list(nltk.bigrams(candidate_dict[date]['word_tokens']))))
        # break

    return candidate_dict

# Returns decoded string for the body of the email
def get_body(message):
    body = None
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        body = subpart.get_payload(decode=True)
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
    elif message.get_content_type() == 'text/plain':
        body = message.get_payload(decode=True)

    if body is not None:
        # print("PAYLOAD1: " + str(body))
        # print("PAYLOAD2: " + str(body.decode('UTF-8')))
        return body.decode('UTF-8', errors='replace')
    else:
        return None

def clean_body(payload):
    if payload is not None:
        # Remove links
        temp = re.sub("(<(.*)>)", " ", payload)
        temp = re.sub("((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?", " ", temp).replace("<>", "")

        # Remove usual encodings (ex: "&#xA0;, \r\n. \t) 
        temp = re.sub("&#x?[a-fA-F0-9]+;", " ", temp)
        temp = re.sub("\\r\\n", " ", temp)
        temp = re.sub("(\\t|\\n|\\r)", " ", temp)
        temp = re.sub("\\xa0", " ", temp)
        #temp = re.sub("\\u200", " ", temp)
        temp = re.sub("(\s)+", " ", temp)

        # Remove receiving email address
        temp = re.sub("98xjsmith@gmail.com", " ", temp)
        return temp
    else:
        return payload

if __name__ == '__main__':
    takeout1_mboxes = [path_to_takeout1 + "/" + f for f in listdir(path_to_takeout1)]
    takeout2_mboxes = [path_to_takeout2 + "/" + f for f in listdir(path_to_takeout2)]

    # print("Looking at mbox: " + str(takeout1_mboxes[0]))
    scrape(takeout1_mboxes[0])
