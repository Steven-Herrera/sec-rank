import tensorflow as tf
import numpy as np
import copy
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import os
import google.protobuf.text_format as pbtext
import tqdm

def data_loader(directory=None):
    if directory == None:
        directory = os.getcwd()
    else:
        directory = directory
    path_to_labeled_dir = "Labeled_Data/ExcelParser_Outputs"
    directory_path = os.path.join(directory, path_to_labeled_dir)
    file_list = os.listdir(directory_path)
    df_dict = {}
    rule_dict = {}
    package = {}
    rule_names = []
    for i in range(len(file_list)):
        if "RuleText" in file_list[i]:
            df = pd.read_csv(os.path.join(directory_path, file_list[i]))
            ruletext = encode_checker(df["Text"].iloc[0])
            rulename = file_list[i].replace("RuleText.csv", "")
            rule_dict[rulename] = ruletext
        if ("RuleText" not in file_list[i]) and (file_list[i][-3:] == ".h5"):
            df = pd.read_hdf(os.path.join(directory_path, file_list[i]))
            rulename = file_list[i][:-3]
            df_dict[rulename] = df
            rule_names.append(rulename)
        elif ("RuleText" not in file_list[i]) and (file_list[i][-3:] == "csv"):
            df = pd.read_csv(os.path.join(directory_path, file_list[i]))
            rulename = file_list[i][:-4]
            df_dict[rulename] = df
            rule_names.append(rulename)

    package["Rules"] = rule_dict
    package["Dataframes"] = df_dict

    # package structure
    # package = { 'Rules': { 'RuleName': str },
    #             'Dataframes': { 'RuleName': pandas df }}

    return package, rule_names

# some words are represented as byte strings that cannot be encoded as utf-8
# this function removes those words
def encode_checker(txt):
    all_words = txt.split()
    can_be_encoded = []
    cannot_be_encoded = {}
    for i in range(len(all_words)):
        word = all_words[i]
        try:
            word.encode("utf-8")
            can_be_encoded.append(word)
        except UnicodeEncodeError:
            cannot_be_encoded[i] = word
    if len(can_be_encoded) == 0:
        return cannot_be_encoded
    elif len(can_be_encoded) > 0 and len(cannot_be_encoded) > 0:
        new_text = " ".join(can_be_encoded)
        return new_text, cannot_be_encoded
    else:
        return new_text