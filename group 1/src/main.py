print("Start")
# from posTagEnricher import getConditionedPrefixes, getSCPrefixes
from wordmodel import loadModel, getNearestNeighbours
import re
import fasttext
import matplotlib.pyplot as plt
import functools


#! USING RAW classifier_model IN PATH, CHANGE IF WANT ORIGINAL ONE!
classifier_model_path = "Data/models/best_classifier.bin"
CANON_NOUNS = "Data/testing/canonNouns.txt"
print("Classifier(s) loading...")
testMode = False

WORDMODEL = loadModel(type="gensim")

NN_Counts = {
    "NC": 0,
    "SC": 0,
    "OC": 0,
    "absPro": 0,
    "possC": 0,
    "adjPre": 0
}
# PREFIX DICTIONARIES--------------------
match_NCprefixes = {
    "umu": False,
    "um": False,
    "u": False, #["1a","11","14"],
    "mu": False,
    "aba":"2",
    "abe":"2",
    "ba": "2",
    "be": "2", 
    "o": "2a", # Make false bc can have o for 1 e.g. ofunday? Technicall
    "bo": "2a", # simple prefix form
    "imi": "4",
    "im": False, #"4", i (NC5) + m
    "mi": "4", # simple prefix2
    "ili": "5",
    "il": "5",
    "li": "5",
    "i": False,
    "ama": "6",
    "am": "6",
    "ma": "6", # simple prefix
    "isi": "7", 
    "is": False, #"7", i (NC5) + s, especially in cases of s-initiating loanwords, e.g. i-softhiwe, i-selula
    "si": "7", # simple prefix
    "izi": False,
    "iz": False,
    "zi": "8",
    "in": False, #"9", i (NC5) + n, e.g. ingovu
    "n": "9",
    "m" : "9", 
    "im": False, #"9", i (NC5) + m, e.g. imayonnaise
    "izim":False, #"14", izi (NC8) + consonant
    "izin":False, #"10", izi (NC8) + consonant
    "zin": "10", # simple
    "zim": "10", # simple
    "lu":"11",  # simple
    "ulu": "11",
    "ul": False, #"11", u (NC1a,3 etc) + consonant
    "ubu": False, #"14", u (NC1a,3 etc) + consonant
    "bu": "14", # simple
    "ub": False, #"14", u (NC1a,3 etc) + consonant
    "uku": "15",
    "ku": "15", # simple
    "uk": False, #"15", u (NC1a,3 etc) + consonant
    "pha": "16", # simple
    "ph": "16",  
}
match_NCprefixes = dict(sorted(match_NCprefixes.items(), key=lambda item: len(item[0]), reverse=True))
NC_prefixes = {'1': ['umu', 'umw', 'um', 'um', "mu","m", "mw"], '1a': ['u', 'w'], '2': ['aba', 'abe', 'abo', 'abe', 'ba', 'ab', 'bo', 'be', 'b'], '2a': ['o', "bo"], '3': ['umu', 'umw', 'um', 'u', 'w',"mu", "m"], '4': ['imi', 'imy', 'im', "mi", "my"], '5': ['i', 'y', "ili", "li"], '6': ['ame', 'ama', 'amo', 'ame', 'am'], '7': ['isi', 'isy', 'is'], '8': ['izi', 'izy', 'iz'], '9': ['im', 'in'], '10': ['izim', 'izin', 'izi', 'izi', 'izy', 'izy', 'iz', 'iz'], '11': ['ulu', 'ulw', 'ul', 'u', 'w'], '14': ['ubu', 'ubw', 'ub', 'u', 'w'], '15': ['uku', 'ukw', 'uk'], '17': ['uku', 'ukw', 'uk']}
SC_prefixes=  {'1': ['aka', 'ako', 'ake', 'ako', 'ake', 'ka', 'wa', 'ko', 'ke', 'wo', 'we', 'ak', 'ak', 'ko', 'ke', 'wo', 'we', 'u', 'w', 'k', 'w', 'k', 'w', 'w'], 
               '1a': ['ka', 'ko', 'ke', 'ko', 'ke', 'u', 'e', 'w', 'k', 'k', 'w'], 
               '2': ['aba', 'abo', 'abe', 'abo', 'abe', 'ba', 'be', 'bo', 'be', 'ab', 'ab', 'bo', 'be', 'b'], 
               '2a':['a', "ab", "aba"], '3': ['aka', 'awu', 'ako', 'ake', 'ako', 'ake', 'wa', 'ak', 'aw', 'wo', 'we', 'ak', 'aw', 'wo', 'we', 'u', 'w', 'w', 'w', 'w'], 
               '4': ['ayi', 'ayy', 'ayy', 'ya', 'ay', 'yo', 'ye', 'ay', 'yo', 'ye', 'i', 'y', 'y', 'y', 'y'], 
               '5': ['ali', 'aly', 'aly', 'li', 'la', 'ly', 'al', 'lo', 'le', 'al', 'ly', 'lo', 'le', 'l', 'l', 'l', 'l'], 
               '6': ['awa', 'awo', 'awe', 'awo', 'awe', 'aw', 'aw', 'a', 'o', 'e', 'o', 'e'], '7': ['asi', 'asy', 'asy', 'si', 'sa', 'sy', 'as', 'so', 'se', 'as', 'sy', 'so', 'se', 's', 's', 's', 's'], 
               '8': ['azi', 'azy', 'azy', 'zi', 'za', 'zy', 'az', 'zo', 'ze', 'az', 'zy', 'zo', 'ze', 'z', 'z', 'z', 'z'], '9': ['ayi', 'ayy', 'ayy', 'ya', 'ay', 'yo', 'ye', 'ay', 'yo', 'ye', 'i', 'y', 'y', 'y', 'y'], 
               '10': ['azi', 'azy', 'azy', 'zi', 'za', 'zy', 'az', 'zo', 'ze', 'az', 'zy', 'zo', 'ze', 'z', 'z', 'z', 'z'], 
               '11': ['alu', 'lwa', 'alw', 'lwo', 'lwe', 'alw', 'lwo', 'lwe', 'lu', 'lw', 'al', 'lw', 'al', 'lw', 'lw', 'l', 'l'], 
               '14': ['abu', 'aba', 'abw', 'abo', 'abe', 'abw', 'abo', 'abe', 'bu', 'ba', 'bw', 'bo', 'be', 'ab', 'ab', 'ab', 'ab', 'bw', 'bo', 'be', 'b', 'b', 'b', 'b'], 
               '15': ['aku', 'kwa', 'akw', 'kwo', 'kwe', 'akw', 'kwo', 'kwe', 'ku', 'kw', 'ak', 'kw', 'ak', 'kw', 'kw', 'k', 'k'], 
               '16': ['kwa', 'aku', 'kwo', 'kwe', 'akw', 'kwo', 'kwe', 'akw', 'ku', 'kw', 'kw', 'ak', 'kw', 'ak', 'kw', 'k', 'k'],
               '17': ['aku', 'kwa', 'akw', 'kwo', 'kwe', 'akw', 'kwo', 'kwe', 'ku', 'kw', 'ak', 'kw', 'ak', 'kw', 'kw', 'k', 'k']} 

OC_prefixes = {'1': ['mu', 'mw', 'm'], '1a': ['mu', 'mw', 'm'], '2': ['ba', 'bo', 'be', 'b'], '2a': ['ba', 'bo', 'be', 'b'], '3': ['wu', 'ww', 'w'], '4': ['yi', 'yy', 'y'], '5': ['li', 'ly', 'l'], '6': ['wa', 'wo', 'we', 'w'], '7': ['si', 'sy', 's'], '8': ['zi', 'zy', 'z'], '9': ['yi', 'yy', 'y'], '10': ['zi', 'zy', 'z'], '11': ['lu', 'lw', 'l'], '14': ['bu', 'bw', 'b'], '15': ['ku', 'kw', 'k'], '17': ['ku', 'kw', 'k']}
ABS_prefixes =  {'1': ['yena', 'yeno', 'yene', 'yen'], '1a': ['bona', 'bono', 'bone', 'bon'], '2': ['yena', 'yeno', 'yene', 'yen'], '2a': ['bona', 'bono', 'bone', 'bon'], '3': ['wona', 'wono', 'wone', 'won'], '4': ['yona', 'yono', 'yone', 'yon'], '5': ['lona', 'lono', 'lone', 'lon'], '6': ['wona', 'wono', 'wone', 'won'], '7': ['sona', 'sono', 'sone', 'son'], '8': ['zona', 'zono', 'zone', 'zon'], '9': ['yona', 'yono', 'yone', 'yon'], '10': ['zona', 'zono', 'zone', 'zon'], '11': ['lona', 'lono', 'lone', 'lon'], '14': ['bona', 'bono', 'bone', 'bon'], '15': ['khona', 'khono', 'khone', 'khon'], '17': ['khona', 'khono', 'khone', 'khon']}
POSS_prefixes = {'1': ['wa', 'ka', 'wo', 'we', 'ko', 'ke', 'w', 'k'], '1a': ['wa', 'ka', 'wo', 'we', 'ko', 'ke', 'w', 'k'], '2': ['baka', 'bako', 'bake', 'bak', 'ba', 'bo', 'be', 'b'], '2a': ['baka', 'bako', 'bake', 'bak', 'ba', 'bo', 'be', 'b'], '3': ['wa', 'ka', 'wo', 'we', 'ko', 'ke', 'w', 'k'], '4': ['ya', 'yo', 'ye', 'y'], '5': ['la', 'lo', 'le', 'l'], '6': ['a', 'o', 'e'], '7': ['sika', 'siko', 'sike', 'sik', 'sa', 'so', 'se', 's'], '8': ['zika', 'ziko', 'zike', 'zik', 'za', 'zo', 'ze', 'z'], '9': ['ya', 'yo', 'ye', 'y'], '10': ['zika', 'ziko', 'zike', 'zik', 'za', 'zo', 'ze', 'z'], '11': ['luka', 'luko', 'luke', 'lwa', 'lwo', 'lwe', 'luk', 'lw'], '14': ['buka', 'buko', 'buke', 'buk', 'ba', 'bo', 'be', 'b'], '15': ['kuka', 'kuko', 'kuke', 'kwa', 'kwo', 'kwe', 'kuk', 'kw'], '17': ['kuka', 'kuko', 'kuke', 'kwa', 'kwo', 'kwe', 'kuk', 'kw']}
# ADJ_prefixes = {
# UTILITY FUNCTIONS -----------------------------------------
def get_subwords(word, min_n, max_n):
    min_n = min_n -1
    subwords = set()  # Using a set to avoid duplicate subwords
    word_length = len(word)
    
    # Iterate over all possible subword lengths within the given range
    for length in range(min_n, max_n + 1):
        # Iterate over all possible positions in the word to extract subwords
        for i in range(word_length - length + 1):
            subword = word[i:i + length]
            subwords.add(subword)
        
        # Handle the case where the subword is the entire word
        if length <= word_length:
            subwords.add(word[:length])  # Prefix
            subwords.add(word[-length:]) # Suffix
    
    return list(sorted(subwords))



def checkSubwords(neighbor, prediction):
    """Predicts labels fro a nn's subwords and returns False if NOT INVALID"""
    global classifier_model
    
    
    subwords= get_subwords(neighbor[0], WORDMODEL.wv.min_n, WORDMODEL.wv.max_n)
    subwords_checker = set(classifier_model.get_subwords(neighbor[0])[0])
    
    # else:
    #     subwords= WORDMODEL.
    for subword in subwords:
            label, _ = classifier_model.predict(subword.replace("<", "").replace(">", ""))
            if (subword != neighbor[0]) and (label[0]==prediction):
                return False
    return True


def isBadNeighbour(nearestNeighbour, POS, classNum, subwords=False):
    #* nearestneighbour format: ( ("x",0.9) , NC1)
    nearestNeighbour = nearestNeighbour[0]
    
    # Two possible interpretations of syntactic method
    if subwords:
        return checkSubwords(nearestNeighbour,f"__label__{POS}{classNum}" )
    else:
        # 15 and 17 have same POS sets
        if classNum=="15":
            classNum = "17"
        match POS:
            case("NC"):
                NN_Counts[POS] = NN_Counts[POS] +1
                for affix in NC_prefixes[classNum]:
                    if affix in nearestNeighbour[0]:
                        #print(affix)
                        #print(nearestNeighbour[0].index(affix))
                        return False
            case("OC"):
                NN_Counts[POS] = NN_Counts[POS] +1
                for affix in OC_prefixes[classNum]:
                    if affix in nearestNeighbour[0]:
                        return False
            case("SC"):
                NN_Counts[POS] = NN_Counts[POS] +1
                for affix in SC_prefixes[classNum]:
                    if affix in nearestNeighbour[0]:
                        return False
            case("possC"):
                NN_Counts[POS] = NN_Counts[POS] +1
                for affix in POSS_prefixes[classNum]:
                    if affix in nearestNeighbour[0]:
                        
                        return False
            case("absPro"):
                NN_Counts[POS] = NN_Counts[POS] +1
                for affix in ABS_prefixes[classNum]:
                    if affix in nearestNeighbour[0]:
                        return False
            case("adjPre"):
                #! UPDATE WITH ACTUAL PREFIXES
                return False
        return True

def extractTagAndClass(word):
    # Use regular expression to separate the tag and class
    match = re.match(r"([a-zA-Z]+)(\d+[a|b]?)", word)
    if match:
        tag = match.group(1)
        class_part = match.group(2)
        return tag, class_part
    else:
        return None, None  # Return None if the word doesn't match the expected format

def getNNConcordPrediction(NN):
    global classifier_model
    classifier = classifier_model
    predict = classifier.predict(NN[0], k=1)
    return (NN, predict[0][0].split("__label__")[1])

if testMode:
    print("GetNNConcord: amathuba - ", getNNConcordPrediction("amathuba"))
    print("GetNNConcord: xxxxx - ", getNNConcordPrediction("xxxxx"))

def getGoodNeighbours(nearestNeighbours, syntacticToggledOn=True, useSubwords=False):
    """Filter out neighbours with concords that don't match the concord for the predicted class.
    Args:
        nearestNeighbours (tupe): ("NN", possXX)
    Returns:
        [("NN", XX)]: Neighbour and pos NUMBER -> . DOES NOT RETURN POSSXX, ONLY XX
    """
    # A NN has format: (NN, POSxx)
   
    neighbours = []
    for neighbour in nearestNeighbours:
        pos, num = extractTagAndClass(neighbour[1])
        # check if including syntactic disambiguation
        if syntacticToggledOn:
            
            if not isBadNeighbour(neighbour, pos, num, useSubwords):
                neighbours.append((neighbour[0],num))  
        else: # if syntactic toggled off, all neighbours are good
            neighbours.append((neighbour[0],num))  
    return neighbours
  
def getModalNNClass(nearestNeighbours):
    counter = {'1': 0, '1a':0, '2': 0,'2a':0, 
             '3': 0, '4': 0, '5': 0, '6': 0, 
             '7': 0, '8': 0, '9': 0, '10': 0, 
             '11': 0, '12': 0, '13': 0, '14': 0, 
             '15': 0, '16': 0, '17':0}
    #ave_p = 0
    #count = len(nearestNeighbours)
    if nearestNeighbours != []:
        for neighbour in nearestNeighbours:
            # incremement counter at NC index. 
            counter[str(neighbour[1])] += 1
            #ave_p += nearestNeighbours[0][1]
            
        return max(counter, key=counter.get)#, (ave_p/count)
    return 0


if testMode:
    print("Single mode:",  getModalNNClass([("a",3), ("",3),("", 5)]))
    print("Multiple modes", getModalNNClass([("a",3), ("",3),("", 5), ("",5)]))

# LEGO FUNCTIONS------------------------------------------
def prefixMethod(word):
    """Checks for a matching NC prefix. Does NOT detect non-Nouns.
    Args:
        word (string): Noun with a NC prefix to check
    Returns:
        False | int(NC_class): Returns False if AMBIGOUS or UNMATCHED prefix
    """
    word = word.lower()
    for prefix in match_NCprefixes:
        if word.startswith(prefix):
            return match_NCprefixes[prefix]
    return False

def removeVerbsfromNN(nn:list):
    global classifier_model
    final_list= []
    simple_classifier = fasttext.load_model("Data/models/raw_simplePOS_Classifier.bin")
    for n in  nn:
        # predictions format: ( (label), array)
        # append format: ( (label) ) -> bad, but other functions assume this so leave for now

        prediction = simple_classifier.predict(n[0])
        if  "V" not in prediction[0][0]:
            final_list.append(n)
    return final_list

def semanticMethod(word, wordModel, NNType="classic", NNtopN=100, noVerbs=False):
    global classifier_model
    """Get a list of Nearest Neighbours for a query word.
    Args:
        word (string): query noun
        wordModel (model object): word model 
        NNType (str, optional): annoy | classic NN. Defaults to "classic".
        NNtopN (int, optional):  Number of top N to get. Defaults to "100".

    Returns:
        _type_: _description_
    """
    nn =list(getNearestNeighbours(wordModel,word, NNtopN,NNType))
    #print("OG NN LIst:", nn)
    if noVerbs:
        nn = removeVerbsfromNN(nn)
    if nn:
        semanticStrength_probability = round(functools.reduce(lambda acc, n: acc + n[1], nn,0),2)/len(nn)
        return nn, semanticStrength_probability
    else:
        print(f"Used nothing for {word}")
    return nn,0

def syntacticMethod(nearestNeighbours, toggledOn=True, subWordMethod=True):
    # nnPredictions: ("neighbour"", POSxx) -> Actually a map object iterable
    nnPredictions = list(map(getNNConcordPrediction, nearestNeighbours))
    
    # If syntactic Method toggled off, all neighbours are true neighbours
    true_neighbours = getGoodNeighbours(nnPredictions, syntacticToggledOn=toggledOn, useSubwords=subWordMethod)
    #print("Filtered neighbours: ", true_neighbours)
    mode = getModalNNClass(true_neighbours)
    return mode

def testClassifierMethod(testSet=CANON_NOUNS):
    global classifier_model
    #print("Testing full system with Canon Noun set.")
    count_dict = {"NC"+nc:(0,0) for nc in ["1","1a","2","2a","3","4","5","6","7","8","9","10","11","14","15"]}
    correct = 0
    total = 0
    num_tests = 0
    
    with open(testSet, "r") as testFile:
        for line in testFile:
            #print(line.split(" ", maxsplit=1))
            label, noun = line.split(" ", maxsplit=1)
            prediction = classifier_model.predict(noun.strip().lower())
            label = extractTagAndClass(label.split("__label__")[1])
            num_tests +=1
            
            #print(label, prediction)
            i = "NC"+str(label[1])
            _, prediction = extractTagAndClass(prediction[0][0].split("__label__")[1])
            
            #print(label[1], prediction)
            if label[1] == prediction:
                # Tuple of format: (correct, total)
                # If correct, incremement BOTH correct AND total
                count_dict[i] = ((count_dict[i][0]+1),(count_dict[i][1]+1))
                correct+=1
            else:
             # If incorrect, incremement ONLY total
                count_dict[i] = ((count_dict[i][0]),(count_dict[i][1]+1))
            total +=1
    accuracy = correct/total
    # print(f"Overall Accuracy: {accuracy}")
    count_dict = dict(map((getAve), count_dict.items()))
    # print("By Class Accuracy:")
    # for nc, count in count_dict.items():
    #     print(f"{nc}: {count}")
    count_dict = list(count_dict.values())
    return accuracy, count_dict


def getNounNC(queryNoun, wordModel, NNType, topN, syntacticOn, no_v=False, subwords=False):
    global classifier_model, classifier_model_path
    morph =prefixMethod(queryNoun)
    if morph:
        return morph, 1
    neighbours, probablityStrength = semanticMethod(queryNoun, wordModel,NNType, topN, no_v)
    if (neighbours == []) and no_v:
        print("Empty neighbours for", queryNoun)
        prediction = classifier_model.predict(queryNoun)
        _, prediction = extractTagAndClass(prediction[0][0].split("__label__")[1])

        return prediction, 1
    final_prediction = syntacticMethod(neighbours, toggledOn=syntacticOn, subWordMethod=subwords)
    return final_prediction, probablityStrength

def testPrefixMethod(testSet = CANON_NOUNS):
    print("Testing full system with Canon Noun set.")
    count_dict = {"NC"+nc:(0,0) for nc in ["1","1a","2","2a","3","4","5","6","7","8","9","10","11","14","15"]}
    correct = 0
    total = 0
    # Probability score is an average of all the probability strengths
    probability_score = 0
    with open(testSet, "r") as testFile:
        for line in testFile:
            label, noun = line.split(" ", maxsplit=1)
            prediction = prefixMethod(noun.strip().lower())
            label = extractTagAndClass(label.split("__label__")[1])
            i = "NC"+str(label[1])
            if prediction:
                if str(label[1]) == prediction:
                    count_dict[i] = ((count_dict[i][0]+1),(count_dict[i][1]+1))
                    correct+=1
             
                count_dict[i] = ((count_dict[i][0]),(count_dict[i][1]+1))
            total +=1
    accuracy = correct/total
    return accuracy
    
def getAve(a):
    
    if a[1][1] > 0:
        return (a[0], str(round(a[1][0]/a[1][1], 2)))
    return (str(a[0]),0)

def testSystem(topN_NN,useSyntactic, testSet = CANON_NOUNS,returnClassAccuracies=False, noVerbs =False, useSubwords = True):
    global classifier_model
    print("Testing full system with Canon Noun set.")
    count_dict = {"NC"+nc:(0,0) for nc in ["1","1a","2","2a","3","4","5","6","7","8","9","10","11","14","15"]}
    correct = 0
    total = 0
    num_tests = 0
    # Probability score is an average of all the probability strengths
    probability_score = 0
    with open(testSet, "r") as testFile:
        for line in testFile:
            label, noun = line.split(" ", maxsplit=1)
            prediction, probability_strength = getNounNC(noun.strip().lower(), WORDMODEL, "classic", topN_NN, syntacticOn =useSyntactic,no_v= noVerbs, subwords=useSubwords)
            label = extractTagAndClass(label.split("__label__")[1])
            probability_score += probability_strength
            num_tests +=1
            i = "NC"+str(label[1])
            if str(label[1]) == prediction:
                # Tuple of format: (correct, total)
                # If correct, incremement BOTH correct AND total
                count_dict[i] = ((count_dict[i][0]+1),(count_dict[i][1]+1))
                correct+=1
            else:
                count_dict[i] = ((count_dict[i][0]),(count_dict[i][1]+1))
            total +=1
    accuracy = correct/total
    probability_score= probability_score/num_tests
    #print(f"Overall Accuracy: {accuracy}")
    #count_dict = dict(map(lambda a: ( a[0], round(a[1][0]/a[1][1], 2) ), count_dict.items()))
    count_dict = (dict(map(getAve, count_dict.items())))
    
    if returnClassAccuracies:
        # accuracy: int, class acc: [value, value...]
        return accuracy, list(count_dict.values())
    # for item, value in count_dict.items(): 
    #     print(item, ":", value)
    return accuracy, probability_score


def getAllClassifierAccuracy_OneKNN(quantity):
    """Gets accuracy for all 6 classifiers for 1 NN quantity

    Args:
        quantity (int): amount of NN to use

    Returns:
        accuracies [list]: List with each classifier's accuracy for given NN amount
    """
    global classifier_model
    models = ["Data/models/raw_full_sentence.bin", # full sen: NC + SC
              "Data/models/raw_sentence.bin", # partial sen: NC + SC
              "Data/models/test_raw_words.bin", # words: NC + SC
              "Data/models/raw_noun_words.bin", # words: only NC
              "Data/models/best_classifier.bin", # words: full tagset -> ann
              "Data/models/nc_sc_data_classifier.bin" # words: nc + sc -> ann
    ]
    accuracies = []
    for m in models:
        print(f"Model: {m}, KNN: {quantity}")
        classifier_model = fasttext.load_model(m)
        totAccuracy, _= testSystem(topN_NN=quantity, useSyntactic=True)
        accuracies.append(round(totAccuracy*100, 2))
    return accuracies

    
def plotNNAccuracies(figName):
    print("Graphing NN accuracies...")
    topN = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    acc = [[],[],[],[],[],[]]
    #probabilites = []
    
    for quantity in topN:
        print(f"Getting accuracy for {quantity}")
        c_accuracy= getAllClassifierAccuracy_OneKNN(quantity)   
        for i in range(6):
            acc[i].append(c_accuracy[i])
    
    col = ["darkgreen",  "lime","orange","darkorchid","red","blue"]
    labels = ['Auto: SC + NC, Full sentence', #x
              'Auto: SC + NC, N + V sentence', #x
              'Auto: SC + NC, words', #v
              'Auto: NC, words', #o
              'Annot.: full set, words', #*
              'Annot.: SC + NC, words'] #v
    lines = ["solid","solid","solid","solid","dashed","dashed"]
    markers = ["x","x", "v", "o","*","v" ]
    
    for i in range(6):
        print(f"Plotting {topN} nn for {labels[i]}")
        plt.plot(topN, acc[i], label=labels[i], color = col[i], linestyle = lines[i], marker = markers[i])
     
    plt.xlabel('topK NN used')
    plt.ylabel('Accuracy by Classifier')
    plt.title('Change in performance over KNN used, by classifier type')
    plt.legend()
    print("Done!")
    plt.savefig(f"Data/results/{figName}.png")
    plt.show()
    plt.close()
    print("Graph closed.")

def recordBestWorstClassAccuracy():
    result = "Data/results/best_worst_classAccuracy_byKNN.csv"
    print(f"Writing best and worst class accuracies to {result}")

    labels = ['Auto: SC + NC, Full sentence', #x
        'Auto: SC + NC, N + V sentence', #x
        'Auto: SC + NC, words', #v
        'Auto: NC, words', #o
        'Annot.: full set, words', #*
        'Annot.: SC + NC, words'] #v
    # hard coded, recorded from plotNNAccuracies() experiment
    
    best = [20, 180, 140, 25, 170, 100]
    worst = [10, 20, 140, 10, 50, 150]
    
    global classifier_model
    models = ["Data/models/raw_full_sentence.bin", # full sen: NC + SC
              "Data/models/raw_sentence.bin", # partial sen: NC + SC
              "Data/models/test_raw_words.bin", # words: NC + SC
              "Data/models/raw_noun_words.bin", # words: only NC
              "Data/models/best_classifier.bin", # words: full tagset -> ann
              "Data/models/nc_sc_data_classifier.bin" # words: nc + sc -> ann
    ]
    with open(result, "w") as outfile: 
        # For each model, use best and worst and write class accuracies on line
        # line1: best
        # line2: worst
        
        # first write headers
        outfile.write("classifier;NNType;KNN;1;1a;2;2a;3;4;5;6;7;8;9;10;11;14;15")
        # then start writing accuracies
        lineContent = ""
        
        for i in range(6):
            classifier_model = fasttext.load_model(models[i])
            # line 1: best
            _, byClassAcc= testSystem(topN_NN=best[i], useSyntactic=True, returnClassAccuracies=True)
            lineContent = labels[i] +";best;" + str(best[i])+(";".join(map(str,byClassAcc[:-1])))
            outfile.write(lineContent+"\n") # Write best line to file
            
            # line 2: worst
            _, byClassAcc= testSystem(topN_NN=worst[i], useSyntactic=True, returnClassAccuracies=True)
            lineContent = labels[i] +";worst;"+str(worst[i])+(";".join(map(str,byClassAcc[:-1])))
            outfile.write(lineContent+"\n") # Write best line to file
            lineContent = ""
        print(f"Completed for {models[i]}")
    print("All done!")
            
def recordAllAccuracies():
    global classifier_model
    
    result = "Data/results/allAccuracies.csv"
    classResult = "Data/results/allAccuracies_byClass.csv"
    print(f"Writing all accuracies to {result}")
    
    testSets = ["Data/testing/canonNouns.txt", "Data/testing/canonNouns_Nicky.txt"]
    systemVersion = [("M0", "Morph Only"),  
                     ("A1","Classifier only"), 
                     ("A2", "...with Semantic"),
                     ("A3","...with No Verbs" ),
                     ("A4","... with Syn., NO Verbs"),
                     ("A5", "...with Syn., AND Verbs"),
                     ("D0","Syn-Sem Diff")]
    
    classifiers = [('Auto: SC + NC, Full sentence',"Data/models/raw_full_sentence.bin", 20), #x
        ('Auto: SC + NC, N + V sentence',"Data/models/raw_sentence.bin",180), #x
        ('Auto: SC + NC, words',"Data/models/test_raw_words.bin", 140), #v
        ('Auto: NC, words',"Data/models/raw_noun_words.bin",25) ,#o
        ('Annot.: full set, words', "Data/models/best_classifier.bin", 170), #*
        ('Annot.: SC + NC, words',"Data/models/nc_sc_data_classifier.bin", 100) ]#v
    
    with open(result, "w") as outfile, open(classResult, "w") as class_out :
        outfile.write("Iteration;Type;Classifier;TestSet1;TestSet2")
        class_out.write("Iteration;Type;TestSet;1a;1;2;2a;3;4;5;6;7;8;9;10;11;14;15;Total")
        for code, version in systemVersion:
            match(code):
                case "M0":
                        mAccT1 = testPrefixMethod(testSet=testSets[0])
                        mAccT2 = testPrefixMethod(testSet=testSets[1])
                        outfile.write(f"{code};{version};None;{str(mAccT1)};{str(mAccT2)}\n")
                case "A1":
                    # Classifier only
                        for name, modelPath, _ in classifiers:
                            print(f"Doing {version} for {name}")
                            classifier_model = fasttext.load_model(modelPath)
                            t0, t0Class =testClassifierMethod(testSets[0]) #classifier_model.test(testSets[0])
                            t1, t1Class =testClassifierMethod(testSets[1]) #classifier_model.test(testSets[1])
                            outfile.write(f"{code};{version};{name};{ str(t0) };{ str(t1) }\n")
                            class_out.write(f"{code};{version};Set1;{';'.join(t0Class)}\n")
                            class_out.write(f"{code};{version};Set2;{';'.join(t1Class)}\n")
                case "A2":
                    # With semantic 
                        for name, modelPath, knn in classifiers:
                            print(f"Doing {version} for {name}")
                            classifier_model = fasttext.load_model(modelPath)
                            t0,t0Class = testSystem(topN_NN=knn, useSyntactic=False, testSet=testSets[0], returnClassAccuracies=True)
                            t1, t1Class= testSystem(topN_NN=knn, useSyntactic=False, testSet=testSets[1], returnClassAccuracies=True)
                            outfile.write(f"{code};{version};{name};{ str(t0) };{ str(t1) }\n")
                            class_out.write(f"{code};{version};Set1;{';'.join(t0Class)}\n")
                            class_out.write(f"{code};{version};Set2;{';'.join(t1Class)}\n")
                            
                case "A3":
                    # Semantic + no verbs
                        for name, modelPath, knn in classifiers:
                            print(f"Doing {version} for {name}")
                            classifier_model = fasttext.load_model(modelPath)
                            t0,t0Class = testSystem(topN_NN=knn, useSyntactic=False, testSet=testSets[0],noVerbs=True, returnClassAccuracies=True)
                            t1, t1Class= testSystem(topN_NN=knn, useSyntactic=False, testSet=testSets[1],noVerbs=True,returnClassAccuracies=True)
                            outfile.write(f"{code};{version};{name};{ str(t0) };{ str(t1) }\n")
                            class_out.write(f"{code};{version};Set1;{';'.join(t0Class)}\n")
                            class_out.write(f"{code};{version};Set2;{';'.join(t1Class)}\n")
                case "A4":
                    # Semantic + no verbs + syntactic
                        for name, modelPath, knn in classifiers:
                            print(f"Doing {version} for {name}")
                            classifier_model = fasttext.load_model(modelPath)
                            t0,t0Class = testSystem(topN_NN=knn, useSyntactic=True, testSet=testSets[0],noVerbs=True, returnClassAccuracies=True)
                            t1, t1Class= testSystem(topN_NN=knn, useSyntactic=True, testSet=testSets[1],noVerbs=True, returnClassAccuracies=True)
                            outfile.write(f"{code};{version};{name};{ str(t0) };{ str(t1) }\n")
                            class_out.write(f"{code};{version};Set1;{';'.join(t0Class)}\n")
                            class_out.write(f"{code};{version};Set2;{';'.join(t1Class)}\n")
                case "A5":
                    # Semantic + VERBS + syntactic
                        for name, modelPath, knn in classifiers:
                            print(f"Doing {version} for {name}")
                            classifier_model = fasttext.load_model(modelPath)
                            t0,t0Class = testSystem(topN_NN=knn, useSyntactic=True, testSet=testSets[0], returnClassAccuracies=True) # NoV - False
                            t1, t1Class= testSystem(topN_NN=knn, useSyntactic=True, testSet=testSets[1], returnClassAccuracies=True) # NoV - False
                            outfile.write(f"{code};{version};{name};{ str(t0) };{ str(t1) }\n")
                            class_out.write(f"{code};{version};Set1;{';'.join(t0Class)}\n")
                            class_out.write(f"{code};{version};Set2;{';'.join(t1Class)}\n")
                #case "B0":
                    # Identify best setting and best final accuracy
                    # plot classifier only final accuracy
                    # plot syntactic final accuracy
                    # so no additional tests!
                    #print("diff part")
                    #* This part, do seperate
    
                # case "A6":
                #     # Couldn't do retrofitting
                #     continue

    print(f"Done. Results written to {result}")


def doExperiment():
    print(f"\033[35m classifier only\033[0m:")
    testClassifierMethod()
    print("\033[35m semantic only\033[0m:")
    testSystem(topN_NN=110, useSyntactic = False)
    print("\033[35m semantic and syntactic (full system):\033[0m:")
    testSystem(topN_NN=110, useSyntactic = True)

def main():
    #print("Neighbour filter:", getGoodNeighbours([("amathuba","NC6"), ("xxx", "SC6")], syntacticToggledOn=False))
    #print("Neighbour filter:", getGoodNeighbours([("amathuba","NC6"), ("xxxX", "SC6")]))
    # print("Testing")
    # nn = [('umamumfundisi', 0.9634546041488647), ('umkamfundisi', 0.9387812614440918), ('wumfundisi', 0.9130711555480957), ('10mfundisi', 0.9073805212974548), ('omfundisi', 0.8996861577033997), ('urnfundisi', 0.8985541462898254), ('umfundisl', 0.8978737592697144), ('umfundis', 0.8975775241851807), ('mamumfundisi', 0.8951287269592285), ('nomamumfundisi', 0.8837154507637024)]
    
    #print(getNounNC("izilungiselelo", wordModel=WORDMODEL, NNType="classic", topN=10))
    global classifier_model, classifier_model_path

    # #! FULL SENTENCES
    classifier_model = fasttext.load_model("Data/models/raw_full_sentence.bin")
    print(f"\033[31mFull sentence level, Classifier:")
    print(classifier_model.test(CANON_NOUNS))
    print(testSystem(200, True, CANON_NOUNS, returnClassAccuracies=False, noVerbs=False, useSubwords=True))
    
    # #! SENTENCES
    # classifier_model = fasttext.load_model("Data/models/raw_sentence.bin")
    # print(f"\033[31mSentence level, Classifier:")
    # print(classifier_model.test(CANON_NOUNS))
    # doExperiment()
    
    # #! WORDS
    # classifier_model = fasttext.load_model("Data/models/raw_words.bin")
    # print(f"\033[31mWord level, Classifier:")
    # print(classifier_model.test(CANON_NOUNS))
    # doExperiment()
    
    # #! NOUNS ONLY
    # classifier_model = fasttext.load_model("Data/models/raw_noun_words.bin")
    # print(f"\033[31mWord level (nouns only)")
    # print(classifier_model.test(CANON_NOUNS))
    # doExperiment()
    
    # # #! Annotated data
    # classifier_model = fasttext.load_model("Data/models/best_classifier.bin")
    # print(f"\033[31mWith annotated data")
    # print(classifier_model.test(CANON_NOUNS))
    # doExperiment()
    classifier_model = fasttext.load_model("Data/models/test_raw_words.bin")
    print(f"\033[31mtest from earlier")
    print(classifier_model.test(CANON_NOUNS))
    doExperiment()
    
    classifier_model = fasttext.load_model("Data/models/nc_sc_data_classifier.bin")
    print(f"\033[31mAnnotated n + cs, Classifier:")
    print(classifier_model.test(CANON_NOUNS))
    doExperiment()


classifier_model = fasttext.load_model("Data/models/raw_sentence.bin")
print(f"\033[31mFull sentence level, Classifier:")
print(testSystem(200, True, CANON_NOUNS, returnClassAccuracies=False, noVerbs=False, useSubwords=True))
print("Normal syntactic:")
print(testSystem(200, True, CANON_NOUNS, returnClassAccuracies=False, noVerbs=False, useSubwords=False))

classifier_model = fasttext.load_model("Data/models/nc_sc_data_classifier.bin")
print(f"\033[31mAnnotated n + cs, Classifier:")
print(classifier_model.test(CANON_NOUNS))
print(testSystem(100, True, CANON_NOUNS, returnClassAccuracies=False, noVerbs=False, useSubwords=True))
print("Normal syntactic:")
print(testSystem(100, True, CANON_NOUNS, returnClassAccuracies=False, noVerbs=False, useSubwords=False))

# testSets = ["Data/testing/canonNouns.txt", "Data/testing/canonNouns_Nicky.txt"]

# classifier_model = fasttext.load_model("Data/models/raw_sentence.bin")

# print("Classifier only:")
# print(testClassifierMethod(testSet=testSets[1]))
# print("Normal:")
# print(testSystem(topN_NN=180, useSyntactic=False, testSet=testSets[0], noVerbs=False))
# print("With No Verbs:")
# print(testSystem(topN_NN=180, useSyntactic=False, testSet=testSets[0], noVerbs=True))
# print("With No Verbs and Syntactic:")
# print(testSystem(topN_NN=180, useSyntactic=True, testSet=testSets[1], noVerbs=True)) 
# print("With YES Verbs and Syntactic:")
# print(testSystem(topN_NN=180, useSyntactic=True, testSet=testSets[1], noVerbs=False))  
# classifier_model = fasttext.load_model("Data/models/nc_sc_data_classifier.bin")
# nn =[('mfula', 0.8677913546562195), ('kumfula', 0.836467444896698), ('nomfula', 0.7915710210800171), ('somfula', 0.7794511914253235), ('komfula', 0.7680025100708008), ('zomfula', 0.7498027682304382),
# ('umgodi', 0.743040919303894), ('umfu', 0.7428905963897705), ('itafula', 0.7370855808258057), ('lomfula', 0.733579695224762)]
# for item in nn:
#     print(classifier_model.predict(item[0]))
# classifier_model = fasttext.load_model("Data/models/raw_sentence.bin")
# print()
# for item in nn:
#     print(classifier_model.predict(item[0]))

