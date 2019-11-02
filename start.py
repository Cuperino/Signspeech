#!/usr/bin/python
# -*- coding: utf-8 -*-

# Signspeech
# Copyright (C) 2019 Javier O. Cordero Pérez <javier@imaginary.tech>.

# This file is part of Signspeech.

# Signspeech is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Signspeech is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Signspeech.  If not, see <https://www.gnu.org/licenses/>.

print ("""
╔═╗╦╔═╗╔╗╔╔═╗╔═╗╔═╗╔═╗╔═╗╦ ╦
╚═╗║║ ╦║║║╚═╗╠═╝║╣ ║╣ ║  ╠═╣
╚═╝╩╚═╝╝╚╝╚═╝╩  ╚═╝╚═╝╚═╝╩ ╩

Signspeech  Copyright (C) 2019  Javier O. Cordero Pérez

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions; check LICENSE for details.

  ┬  ┌─┐┌─┐┌┬┐┬┌┐┌┌─┐  ┬  ┬┌┐ ┬─┐┌─┐┬─┐┬┌─┐┌─┐
  │  │ │├─┤ │││││││ ┬  │  │├┴┐├┬┘├─┤├┬┘│├┤ └─┐
  ┴─┘└─┘┴ ┴─┴┘┴┘└┘└─┘  ┴─┘┴└─┘┴└─┴ ┴┴└─┴└─┘└─┘
""")

# Imports
import azure.cognitiveservices.speech as speechsdk
import os
import subprocess
import stanfordnlp

# Download models on first run
# stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
# Sets up a neural pipeline in English
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', treebank='en_ewt', use_gpu=False, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size

def getSpeech():
  # Creates an instance of a speech config with specified subscription key and service region.
  # Replace with your own subscription key and service region (e.g., "westus").
  with open('../../keys/speech_key.txt','r') as f_open:
      speech_key = f_open.read()
      f_open.close()
  with open('../../keys/speech_region.txt','r') as f_open:
      service_region = f_open.read()
      f_open.close()

  # Creates an instance of a speech config with specified subscription key and service region.
  # Replace with your own subscription key and service region (e.g., "westus").
  speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

  # Creates a recognizer with the given settings
  speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

  print("Say something to translate...")

  # Starts speech recognition, and returns after a single utterance is recognized. The end of a
  # single utterance is determined by listening for silence at the end or until a maximum of 15
  # seconds of audio is processed.  The task returns the recognition text as result. 
  # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
  # shot recognition like command or query. 
  # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
  result = speech_recognizer.recognize_once()

  # Checks result.
  if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Recognized: {}\n".format(result.text))
  elif result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech could be recognized: {}".format(result.no_match_details))
    quit()
  elif result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = result.cancellation_details
    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
      print("Error details: {}".format(cancellation_details.error_details))
    quit()

  return result.text

def parse(text):
  # Process text input
  doc = nlp(text) # Run the pipeline on text input
  print("\n")

  # Look at the results
  for sentence in doc.sentences:
    for word in sentence.words:
      print(word)
  # doc.sentences[0].print_tokens() # Look at the result
  # doc.sentences[0].print_dependencies()
  
  return doc.sentences

def reorder(parse):
  aslTree = parse

  return aslTree

def reduce(tree):
  return tree

def getLemmasOnly(tree):
  translation = []
  # translation = tree
  return translation

def translate(text, parse):
  aslTree = reorder(parse)
  reducedAslTree = reduce(aslTree)
  translation = getLemmasOnly(reducedAslTree)
  return translation

def display(lemmas):
  folder = os.getcwd()
  filePrefix = folder + "/videos/"
  # Alter ASL lemmas to match sign's file names.
  # In production, these paths would be stored at the dictionary's database.
  files = [ filePrefix + lemma + "_.mp4" for lemma in lemmas ]
  # Run video sequence using the MLT Multimedia Framework
  print("Running command: ", ["melt"] + files)
  process = subprocess.Popen(["melt"] + files + [filePrefix + "black.mp4"], stdout=subprocess.PIPE)
  result = process.communicate()

def main():
  # Get text
  print ("""
  ┌─┐┌─┐┌┬┐┬ ┬┌─┐┬─┐  ┌─┐┌─┐┌─┐┌─┐┌─┐┬ ┬
  │ ┬├─┤ │ ├─┤├┤ ├┬┘  └─┐├─┘├┤ ├┤ │  ├─┤
  └─┘┴ ┴ ┴ ┴ ┴└─┘┴└─  └─┘┴  └─┘└─┘└─┘┴ ┴
  """)

  # tests = []
  tests = [
    "What is your name?",
    "Where is the bathroom?",
    "Love computers!"
  ]

  if len(tests) == 0:
    tests = tests + [ getSpeech() ]

  for text in tests:
    print("\nText to process: ", text, "\n")

    print ("""
    ┌─┐┌┐┌┌─┐┬ ┬ ┬┌─┐┌─┐  ┌─┐┌┐┌┌─┐┬  ┬┌─┐┬ ┬
    ├─┤│││├─┤│ └┬┘└─┐├┤   ├┤ ││││ ┬│  │└─┐├─┤
    ┴ ┴┘└┘┴ ┴┴─┘┴ └─┘└─┘  └─┘┘└┘└─┘┴─┘┴└─┘┴ ┴
    """)

    parsed = parse(text)
    print("Parse", parsed)

    print ("""
    ┌─┐┌─┐┬─┐┌─┐┌─┐┬─┐┌┬┐  ┌┬┐┬─┐┌─┐┌┐┌┌─┐┬  ┌─┐┌┬┐┬┌─┐┌┐┌
    ├─┘├┤ ├┬┘├┤ │ │├┬┘│││   │ ├┬┘├─┤│││└─┐│  ├─┤ │ ││ ││││
    ┴  └─┘┴└─└  └─┘┴└─┴ ┴   ┴ ┴└─┴ ┴┘└┘└─┘┴─┘┴ ┴ ┴ ┴└─┘┘└┘
    """)
    
    translation = translate(text, parse)
    translation = ["your", "name", "what"]
    print("\nResult: ", translation, "\n")

    # print ("""
    # ┌─┐┌─┐┬    ┬─┐┌─┐┌─┐┬─┐┌─┐┌─┐┌─┐┌┐┌┌┬┐┌─┐┌┬┐┬┌─┐┌┐┌
    # ├─┤└─┐│    ├┬┘├┤ ├─┘├┬┘├┤ └─┐├┤ │││ │ ├─┤ │ ││ ││││
    # ┴ ┴└─┘┴─┘  ┴└─└─┘┴  ┴└─└─┘└─┘└─┘┘└┘ ┴ ┴ ┴ ┴ ┴└─┘┘└┘
    # """)
    # display(translation)

main()
