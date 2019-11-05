# Signspeech

English to American Sign Language (ASL) translation software prototype developed for an Artificial Intelligence Hackathon.

Signspeech is capable of translating basic sentences from English to American Sign Language (ASL). Speech is gathered using a Speech to Text API. Text is decomposed using the Stanford NLP library and further processed into an ALS syntax. Using the sequence of annotated words, the program plays a sequence of videos using the MLT Multimedia Framework.

[![English to ASL Translation AI - Signspeech (Live Test)](https://img.youtube.com/vi/DlYEgU-REjg/0.jpg)](https://youtu.be/DlYEgU-REjg)


## How to Install

This program is written in Python, you must have Python 3.6 or newer installed on your computer to run it.

### On Ubuntu Linux

`sudo apt-get update`

`sudo apt-get install build-essential libssl1.0.0 libasound2 melt`

`python -m pip install --user azure-cognitiveservices-speech`

`python -m pip install --user stanfordnlp`

From within your Python console run:

`import stanfordnlp`

`stanfordnlp.download('en')`

In order to use the voice feature, your program needs to connect to Microsoft Azure's Speech to Text API. The program can be used without the voice component with simple changes to the source code.

https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/quickstarts/speech-to-text-from-microphone?tabs=dotnet%2Candroid%2Clinux&pivots=programming-language-python

## How to Run

1. `python start.py`
2. Speak a simple sentence to the machine. (Make sure your microphone is on!)

The machine will generate a sequence of ASL words and their lemmas, in English. This sequence will follow a syntactical structure that resembles that of ASL. Then the program will attempt to play back a video sequence with each of the signs. If a video for a sign isn't found, this sign will be skipped, but it can still be found in English at the console's output.

## Gallery

Speech is gathered using a Speech to Text API.

![](https://imaginary.tech/wp-content/uploads/sites/4/2019/11/1.png)

Text is decomposed using the Stanford NLP library and further processed into an ALS syntax.

![](https://imaginary.tech/wp-content/uploads/sites/4/2019/11/2.png)

Using the sequence of annotated words, the program plays a sequence of videos using the Melt Multimedia Framework.

![](https://imaginary.tech/wp-content/uploads/sites/4/2019/11/3.png)

Sign videoclip Copyright by Signing Savvy, LLC. Used for demonstrative educational purposes.

![](https://imaginary.tech/wp-content/uploads/sites/4/2019/11/4.png)

## Disclaimers

This software comes with no videos to generate the ASL output because I couldn't find consistent, high quality, videos of signs that were freely available under Public Domain or a compatible license, such as the Creative Commons License. You may need to add and name your own signs in order to use this program.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
