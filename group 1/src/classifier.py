"""This file is used to train and test FastText classifiers """

import fasttext
import csv
from collections import defaultdict
import re

OG_CANON_FILE = "Data/canonNouns/canonTest.csv"
CLEAN_RAWDATA = "Data/cleaned/rawCleaned.txt"
CANON_DATA = "Data/training/canonNouns.valid"

# Change default classifier training data here
TRAINING_DATA = "Data/training/data.train"
TESTING_DATA = "Data/training/data.valid"
# Change default classifier here
PRETRAINED_MODEL = "Data/models/classifier.bin"


def trainClassifier(modelName, file=TRAINING_DATA, validFile =TESTING_DATA, useVectors = False):
    # Uncomment verbose argument to see the hyperparamters during autotuning
    model = fasttext.train_supervised(input=file,autotuneValidationFile=validFile)#, verbose=5)

    model.save_model("Data/models/"+modelName+".bin")
    print(f"Classifier trained using {file} and saved at Data/models/{modelName}")

def testClassifier(model=PRETRAINED_MODEL, testName="data"):
    model = fasttext.load_model(model)
    result = model.test("Data/training/"+testName+".valid")
    print(result)

def getCanonicalNouns(canonFile=OG_CANON_FILE):
    """Converts file of canonical nouns into a dictionary of Noun: NCx
    Args:
        canonFile (.csv path string)
    Returns:
        dict: noun:NC
    """
    noun_dict = {}
    with open(canonFile, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            noun = row[0]
            noun_class = row[1]
            noun_dict[noun] = "NC"+noun_class           
    return noun_dict

def evaluateClassifier_Canon(model=PRETRAINED_MODEL):
    classifier = fasttext.load_model(model)
    result = classifier.test(CANON_DATA)

    print("\nClassifier testing with CANON NOUNS complete.")
    print(result)

def evaluateClassifier_ALL(modelName, test):
    modelName = "Data/models/"+modelName+".bin"
    testClassifier(model=modelName,testName=test)
    evaluateClassifier_Canon(model=modelName)
    
def getDataIngredients(data):
    # Initialize defaultdict to count occurrences of each label
    label_counts = defaultdict(int)

    with open(data, 'r') as file:
        for line in file:
            words = line.strip().split()
            for word in words:
                if word.startswith('__label__'):
                    label = word.replace('__label__', '')
                    label_counts[label] += 1
                    
    sortedCounts = dict(sorted(label_counts.items()))
    for label, count in sortedCounts.items():
        print(f'{label}: {count}')

def main():
    opt = input("""What'll it be tonight folks?\n 
                (1) Train classifier and test combo \n   
                (2) Load saved classifier and retest combo\n""")
    
    #* Adjust classifier training and testing data, and name to use as input here
    c_train = "Data/training/gold_nn_w.train"
    c_test = "Data/training/gold_nn_w.valid"
    c_name = "gold_nn_w"
    match(int(opt)):
        case 1:
            model = trainClassifier(modelName = c_name, file=c_train, 
                                    validFile=c_test)
            evaluateClassifier_ALL(modelName=c_name, test = c_test)
        case 2:
            evaluateClassifier_ALL(modelM=c_name,test=c_test) 
            
        case _:
            print("Command not recognised. Please close and rerun file.")

