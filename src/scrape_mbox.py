import os, sys, nltk
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
      payload = message.get_payload()      
      #text_payload = clean_payload(str(payload[0]))
      text_payload = getbody(message)

      candidate_dict[date] = {}
      candidate_dict[date]['subject'] = subject
      candidate_dict[date]['payload'] = text_payload

      print("Date: " + str(date))
      print("Subject: " + str(subject))
      print("Payload: " + str(text_payload))
      break

   print(dict(candidate_dict))

def getbody(message): #getting plain text 'email body'
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

   return body

def clean_payload(payload):
   #clean = payload
   clean = payload.replace("(=..)", "")  
   return clean


if __name__ == '__main__':
   takeout1_mboxes = [path_to_takeout1 + "/" + f for f in listdir(path_to_takeout1)]
   takeout2_mboxes = [path_to_takeout2 + "/" + f for f in listdir(path_to_takeout2)]
   
   print("Looking at mbox: " + str(takeout1_mboxes[0]))
   scrape(takeout1_mboxes[0])
