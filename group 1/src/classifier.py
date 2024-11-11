import fasttext
import csv
from collections import defaultdict
import re

OG_CANON_FILE = "Data/canonNouns/canonTest.csv"
CLEAN_RAWDATA = "Data/cleaned/rawCleaned.txt"
RAW_TRAINING = "Data/training/raw.train"
RAW_TESTING = "Data/training/raw.valid"
RAW_MODEL = "Data/models/raw_classifier.bin"
CANON_DATA = "Data/training/canonNouns.valid"
TRAINING_DATA = "Data/training/data.train"
TESTING_DATA = "Data/training/data.valid"
PRETRAINED_MODEL = "Data/models/classifier.bin"
INCORRECT_PREDICTIONS = "Data/incorrect_predictions.txt"


# Best epoch result was 20 (0.299 for validation data, 0.69 for test accuracy)
# Best leanring rate was 0.3
def trainClassifier(modelName, file=TRAINING_DATA, validFile =TESTING_DATA, useVectors = False):
    #model = fasttext.train_supervised(input=file, epoch = epochs, lr = lRate, wordNgrams=wordNG,dim=100, minn=3, maxn=6)
    model = fasttext.train_supervised(input=file,autotuneValidationFile=validFile, verbose=5)

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

# train  = "Data/training/new_sc_nc.train"
# test = "Data/training/new_sc_nc.valid"
# trainClassifier("new_sc_nc_classifier", file = train, validFile=test)
# evaluateClassifier_ALL("new_sc_nc_classifier", "new_sc_nc")
# model = fasttext.load_model("Data/models/new_sc_nc_classifier.bin")
# print(model.test("Data/training/new_sc_nc.valid"))
# # args_obj = model.f.getArgs()
# for hparam in dir(args_obj):
#     if not hparam.startswith('__'):
#         print(f"{hparam} -> {getattr(args_obj, hparam)}")
def main():
    opt = input("""What'll it be tonight folks?\n 
                (1) Retrain Classifier and test combo \n   
                (2) Load saved classifier and retest combo\n
                (3) Raw words and sentences\n""")
    raw_sentences_train = "Data/training/raw_sentence.train"
    raw_sentences_test = "Data/training/raw_sentence.valid"
    raw_words_train = "Data/training/raw_word.train"
    raw_words_test = "Data/training/raw_word.valid"
    nc_sc_train = "Data/training/nc_sc_data.train"
    nc_sc_test = "Data/training/nc_sc_data.valid"
    match(int(opt)):
        case 1:
            
            model = trainClassifier(modelName = "annotated_withVectors", file="Data/training/nc_sc_data.train", 
                                    validFile="Data/training/nc_sc_data.valid", 
                                    useVectors="Data\models\zu_mine-3_6_150.bin")
            evaluateClassifier_ALL(modelM = "Data/models/annotated_withVectors.bin", test = "nc_sc_data")
            evaluateClassifier_ALL(modelM = "Data/models/nc_sc_data_classifier.bin", test = "nc_sc_data")
        case 2:
            savedModel = PRETRAINED_MODEL
            evaluateClassifier_ALL(modelM=RAW_MODEL,test="raw")
        case 3:
            print("USING LABELLED RAW SENTENCES:-------------------------")
            #model = trainClassifier(modelName= "raw_sentence", file=raw_sentences_train, validFile=raw_sentences_test)
            evaluateClassifier_ALL(modelName="raw_sentence",test="raw_sentence")
            
            print("USING LABELLED RAW WORDS:-------------------------")
            #model = trainClassifier(modelName= "raw_words", file=raw_words_train, validFile=raw_words_test)
            evaluateClassifier_ALL(modelName="raw_words",test="raw_word")
            
            raw_full_sentences_train = "Data/training/raw_full_sentence.train"
            raw_full_sentences_test = "Data/training/raw_full_sentence.valid"
            # model = trainClassifier(modelName= "raw_full_sentence", file=raw_full_sentences_train, validFile=raw_full_sentences_test)
            # evaluateClassifier_ALL(modelName="raw_sentence",test="raw_sentence")
            # evaluateClassifier_ALL(modelName="raw_full_sentence",test="raw_full_sentence")
        case _:
            print("Command not recognised.")

def playground(model=PRETRAINED_MODEL):
    model = fasttext.load_model(model)
    label,_=(model.predict("umfundi"))
    print(label[0])   
    # get the class at end of label (__label__XY where Y is number)
    label = re.findall("\d+(?:a|b)?", "<z12>")
    print(label)
    label = re.findall("\d+(a|b)?", "<z12>")
    print(label)
    
# Train word model with 136 dimensions
# Get vocabulary as .vec file
# Train classifier, use 136 dimensions

