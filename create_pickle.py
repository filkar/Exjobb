import csv
import pickle
import sys
import typing
import json
import numpy

from typing import IO
from typing import List
from typing import Dict
from typing import Optional


def preprocess(csv_path: IO) -> List[List[str]]:
    with open(csv_path, "rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)[1:]
        # special_1 = u"\u0091"
        # special_2 = u"\u0092"
        # special_3 = u"\u0097"
        new_list = []
        for line in data:
            line = line[:7]
            # line[1] = line[1].replace(special_1, "")
            # line[1] = line[1].replace(special_2, "\'")
            # line[1] = line[1].replace(special_3, " ")
            new_list.append(line)
        return new_list

def speaker_dict(data: list, last_entry: int) -> dict:
    spkr_dict = {}
    for line in data:
        line_id = int(line[5]) + last_entry
        if line_id not in spkr_dict:
            dialogue_dict = {}
            spkr_dict[line_id] = []
        if line[2] not in dialogue_dict:
            dialogue_dict[line[2]] = len(dialogue_dict)
        speaker_indices = [0] * 9
        speaker_indices[dialogue_dict[line[2]]] = 1
        spkr_dict[line_id].append(speaker_indices)
    return spkr_dict


def emotion_dict(data: list, last_entry: int) -> dict:
    emo_dict = {}
    emotions = {
        "neutral"  : 0,
        "surprise" : 1,
        "fear"     : 2,
        "sadness"  : 3,
        "joy"      : 4,
        "disgust"  : 5,
        "anger"    : 6,
    }
    for line in data:
        line_id = int(line[5]) + last_entry
        if line_id not in emo_dict:
            emo_dict[line_id] = []
        emo_dict[line_id].append(emotions[line[3]])
    return emo_dict

def sentiment_dict(data: list, last_entry: int) -> dict:
    sent_dict = {}
    sentiments = {
        "neutral"  : 0,
        "positive" : 1,
        "negative" : 2,
    }
    for line in data:
        line_id = int(line[5]) + last_entry
        if line_id not in sent_dict:
            sent_dict[line_id] = []
        sent_dict[line_id].append(sentiments[line[4]])
    return sent_dict

def utterance_dict(data: list, last_entry: int) -> dict:
    utter_dict = {}
    for line in data:
        line_id = int(line[5]) + last_entry
        if line_id not in utter_dict:
            utter_dict[line_id] = []
        utter_dict[line_id].append(line[1]) 
    return utter_dict

def dialogue_id_list(data: list, last_entry: int) -> dict:
    id_list = []
    i = -1
    for item in data:
        if int(item[5]) > i:
            id_list.append(int(item[5]) + last_entry)
            i = int(item[5])
        
        
    #id_list = list(range(last_entry, int(data[-1][5]) + last_entry + 1))
    return id_list

# def join_dicts(input_list: List[Dict]) -> str:
#     output_string = "" 
#     output_string += "{"
#     for item in input_list:
#         for key in item:
#             output_string += f"{key}: {item[key]}, "
#     output_string = output_string[:-2]
#     output_string += "}"
#     return output_string
    

def list_output(input_list: List[List]) -> str:
    output_string = ""
    for item in input_list:
        output_string += f"{item} '\n' "
    output_string = output_string.rstrip()
    return output_string

def pickle_files(*pickles):
    with open ("output.pkl", "wb") as file:
        for item in pickles:
            pickle.dump(item, file)

def get_glove() -> dict:
    with open("glove_matrix.txt", "r") as file:
        output_string = file.read()
        output_dict = {}
        output_dict[output_string[1]] = output_string[4:]
    return output_dict

def join_dicts(train: dict, dev: dict, test: dict) -> dict:
    return train | dev | test


def construct_dicts(train: list, dev: list, test: list) -> None:
    pickle = True

    speak_train, speak_dev, speak_test = _full_speaker(train, dev, test)
    emo_train, emo_dev, emo_test = _full_emotion(train, dev, test)
    sent_train, sent_dev, sent_test = _full_sentiment(train, dev, test)
    utter_train, utter_dev, utter_test = _full_utterance(train, dev, test)
    dialogue_id_train, dialogue_id_test, dialogue_id_dev = _full_dialogue_id(train, dev, test) #dialogue_id is in other order in .pkl files given
    
    pickle_files(
        join_dicts(speak_train, speak_dev, speak_test),
        join_dicts(emo_train, emo_dev, emo_test),
        join_dicts(sent_train, sent_dev, sent_test),
        join_dicts(utter_train, utter_dev, utter_test),    
        dialogue_id_train,
        dialogue_id_test,
        dialogue_id_dev,
    )

def _full_speaker(train: list, dev: list, test: list) -> (dict, dict, dict):
    return speaker_dict(train, 0), speaker_dict(dev, int(train[-1][5]) + 1), speaker_dict(test, int(train[-1][5]) + 1 + int(dev[-1][5]) + 1)

def _full_emotion(train: list, dev: list, test: list) -> (dict, dict, dict):
    return emotion_dict(train, 0), emotion_dict(dev, int(train[-1][5]) + 1), emotion_dict(test, int(train[-1][5]) + 1 + int(dev[-1][5]) + 1)

def _full_sentiment(train: list, dev: list, test: list) -> (dict, dict, dict):
    return sentiment_dict(train, 0), sentiment_dict(dev, int(train[-1][5]) + 1), sentiment_dict(test, int(train[-1][5]) + 1 + int(dev[-1][5]) + 1)   

def _full_utterance(train: list, dev: list, test: list) -> (dict, dict, dict):
    return utterance_dict(train, 0), utterance_dict(dev, int(train[-1][5]) + 1), utterance_dict(test, int(train[-1][5]) + 1 + int(dev[-1][5]) + 1)

def _full_dialogue_id(train: list, dev: list, test: list) -> (dict, dict, dict):
    return dialogue_id_list(train, 0), dialogue_id_list(test, int(train[-1][5]) + 1 + int(dev[-1][5]) + 1), dialogue_id_list(dev, int(train[-1][5]) + 1)



def main(train_path: IO, dev_path: IO, test_path: IO) -> (dict, dict, dict):

    train = preprocess(train_path)
    dev = preprocess(dev_path)
    test = preprocess(test_path)

    construct_dicts(train, dev, test)

    
    #print(data)
    #speaker_dict(data)


    
    #print(f"{emotion_dict(data)}")
    #print(f"{sentiment_dict(data)}")
    #print(f"{utterance_dict(data)}")
    
    #dict.keys()[-1]

if __name__ == "__main__":
    main(*sys.argv[1:4]) # pylint: disable = E1120