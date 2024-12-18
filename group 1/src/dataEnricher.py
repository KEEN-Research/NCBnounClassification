from dataPreparer import generateTestTrain
import re
import time
IN_FILE= "Data/source/FullPOS.ZU.cleanedData_Z.txt"
SENTENCE_FILE = "Data/clean_data/raw_sentences_labelled.txt"
WORD_FILE = "Data/clean_data/raw_words_labelled.txt"
FULL_SENTENCE_FILE = "Data/clean_data/raw_full_sentences_labelled.txt"

#* Examples for testing ----------------------------------------
# Verb SC extractor
examples = ["unikeza;3", "ubukele;1","badlulise;2a","babike;2","bekhethwe;2","ikhethe;4",
            "lithuthukisiwe;5", "lokwenza;5", "asebenzisa;6","sihambisane;7","ziye;8",
            "zingokuhlosiwe;10","nokumela;15", "kwenza;15", "kuwenza;15"]

exampleFile = """izindlela	N10
zakhe	POSS10
zithembekile	V
izikhathi	N08
zonke	PROQUANT08

izahlulelo	N08
zakho	POSS10
ziphakeme	V
kakhulu	ADV
kunokubona	V
kwakhe	V

zonke	PROQUANT08
izitha	N08
zakhe	POSS10
uyazidelela	V

izintaba	N10
zathuthumela	POSS10
ebusweni	NLOC
bukajehova	V
yebo	POSS04
nalo	ADV
isinayi	N07
lelo	CDEM05
ebusweni	NLOC
bukajehova	V
unkulunkulu	N01
kaisrayeli	N00

ezinsukwini	REL"""

# Below follows format: [0: SC+C, 1:SC+a/e, 2:SC+o, 3: -SC,4:continuous form,5:other]
# where C is a-z NOT IN {a,e,i,o,u}

sc_prefixes = {
    "1": ["u","w","w","ka","e","a"],
    "1a": ["u","w","w","ka","e","a"],
    "2": ["ba", "b", "b", False,"be", False],
    "2a": ["ba", "b", "b", False,"be", False],
    "3": ["u", "w", "w", "wu", False,False],
    "4": ["i", "y", "y", "yi",False,False],
    "5": ["li","l", "l",False, False, False],
    "6": ["a", "", "", "wa", "e", False],
    "7": ["si", "s", "s", False, False, False],
    "8": ["zi", "z", "z",False, False, False],
    "9": ["i", "y", "y", "yi",False, False],
    "10": ["zi", "z", "z",False, False, False],
    "11": ["lu","lw","l",False,False,False],
    "14": ["bu", "b", "b",False, False, False],
    "15": ["ku","kw", "k",False,False,False],
    "17": ["ku","kw", "k", False,False,False]
}

#* Data enirchment (additional labelling, and data level restructuring) functions
#*--------------------------------------------------------------------------------------
# Returns string OR False
def labelVerbSC(verb:str, classes:list):
    # For +SC only, include neg prefix "se"
    # make regex to check prefix[0], prefix[1], prefix[2]
    # Rule 1 (+): prefix[0]: [ka|ma|a|k|m]?[prefix[0]]+NOT[a|e|i|o|u]
    # Rule 2 (+): prefix[1]: [ka|ma|a|k|m]?[prefix[1][a|e]
    # Rule 3 (+): prefix[2]: [ka|ma|a|k|m]?[prefix[2]][o]
    # Rule 4 (-): prefix[3]: [ka|ma|a|k|m]?a[prefix[2]]
    # Rule 5: prefix[4]: [ka|ma|a|k|m]?be?[prefix[3]]
    # Rule 6: prefix[5]: [ka|ma|a|k|m]?nga?[prefix[4]]
    # Sources:
    # (aw|awu|mawu|ma) -> PPRE in ZuluVerbRules.cfg
    # (se) -> EXCL in zuluverbrules.cfg
    # all other -> oxford isizulu <-> english biligual school dictionary
    for nc in classes:
        #print(classes)
        for i in range(0,6):
            prefix_set = sc_prefixes[str(nc)]
            match(int(i)):
                case(0): # Pos SC check for 4-6 affix in case 4-6 uses 1-3
                    pattern = r"\b(se)?(aw|awu|mawu|ma)?(ka|a|kha|kh|ma|m)?"+prefix_set[i]+"(?![a|e|i|o|u])"

                    if re.match(pattern, verb):
                        return nc
                case(1): # Pos SC + check for 4-6 affix in case 4-6 uses 1-3
                    pattern = r"\b(se)?(aw|awu|mawu|ma|ka|kha|ma|a|be|b)?"+prefix_set[i]+"(?=a|e)"
                    if re.match(pattern, verb):
                        return nc
                case(2): # Pos SC + check for 4-6 affix in case 4-6 uses 1-3
                    pattern = r"\b(se)?(aw|awu|mawu|ma|ka|kha|ma|a|be|b)?"+prefix_set[i]+"(?=o)"
                    if re.match(pattern, verb):
                        return nc
                case(3): # Neg SC -> IF not False. If False, accounted for in 1-3
                    if prefix_set[i]:
                        pattern = r"\b(ka|kha|ma|a|k|kh|m)?"+prefix_set[i] +"()?"
                        if re.match(pattern, verb):
                            return nc
                case(4): #  Cont SC -> IF not False. If False, accounted for in 1-3
                    if prefix_set[i]:
                        pattern = r"\b(ka|kha|ma|a|be|k|kh|m)?"+prefix_set[i]
                        if re.match(pattern, verb):
                            return nc
                case(5): # Subjunctive SC -> IF not False. If False, accounted for in 1-3
                    if prefix_set[i]:
                        pattern = r"\b(ka|kha|ma|a|k|kh|m)?"+prefix_set[i]
                        if re.match(pattern, verb):
                            return nc
    return False

def testLabels():
    classes = []
    t0 = time.time()
    correct = 0
    for example in examples:
        n, c= example.split(";")
        classes.append(c)
        ans = labelVerbSC(n, classes)

        if ans:
            print(str(ans), "   ", c)
            if str(ans) == (c):
                print("correct")
                correct +=1
            else:
                print("incorrect")
        else:
            print(False, " ", c[0])
        classes = []
    print(correct/len(examples))
    t1 = time.time()
    print(f"Took {t1-t0} time")


# Use for sentence labels
def labelAllSentenceVerbs(verbs, classes):
    sc = []
    badVerbs = []
    if verbs:
        for v in verbs:
            v_sc = labelVerbSC(v, classes)
            # If the verb SC matches any sentence noun's SC concord, add to list
            if v_sc:
                sc.append("__label__SC"+str(v_sc))
            else:
                badVerbs.append(v)
        if sc:
            sc = list(set(sc))
            return sc, badVerbs
    return False, badVerbs
        
def isVerb(verb):
  if verb[-2:]=="\tV":
      return True
  return False

def isNoun(noun):
    if "N1" in noun[-3:] or "N0" in noun[-3:] or "N0" in noun[-4:]:
        return True
    return False

def getVerbs(sentence:list):
    return list(filter(isVerb, sentence))

def getLabelledNouns(sentence):
    nouns = []
    for word in sentence:
        if "N1" in word[-3:] or "N0" in word[-3:] or "N0" in word[4:]:
            n, nc = word.split("\t")
            nc = nc[-2:]
            if nc in ["1A", "2A"]:
                noun = ("__label__NC"+nc.lower() + " "+(n)).strip()
                nouns.append(noun)
            else:
                noun = ("__label__NC"+str(int(nc)) + " "+(n)).strip()
                nouns.append(noun)
    return nouns

def getGoodVerbsWithLabel(verbs, classes):
    goodVerbs = []
    if verbs:
        for v in verbs:
            v_sc = labelVerbSC(v, classes)
            # If the verb SC matches any sentence noun's SC concord, add to list
            if v_sc:
                #print("__label__SC"+str(v_sc)+" "+v.split("\t")[0])
                goodVerbs.append("__label__SC"+str(v_sc)+" "+v.split("\t")[0].strip())
    # Will be empty (Falsy) if no goodverbs
    return goodVerbs

def notNoisy(line):
    # seperate word and POS
    parts = (line.strip()).split('\t')
    # Remove lines without a POS
    if len(parts) <2:
        return False
    elif len(parts[0].strip()) <= 4:
            return False
    # Remove items with non-applicable NC's
    elif "00" in parts[1]:
        return False
    
    #* DEFINE LINES WE WANT
    # exclude POSS for now because seems wonkier
    elif parts[1] == "V":
        return True
    elif "N0" in parts[1]:
        return True
    elif "N1" in parts[1]:
        return True
    elif "FOR" in parts[1]:
        return False
    elif "IDEO" in parts[1]:
        return False
    #* Remainders are items we don't want
    # Previously False:
    return False
  
def makeNCLabels(classes):
    nc = []
    for n in classes:
        nc.append("__label__NC"+str(n))
    return " ".join(nc)
      
def getNCfromSentenceNouns(sentence:list):
    nouns = list(filter(isNoun, sentence))
    #print("Noun list: ", nouns)
    ncList = []
    if nouns:
        for n in nouns:
            c = n[-2:]
            if c in ["1A", "2A"]:
                ncList.append(c.lower())
            else:
                ncList.append(int(c))
        return list(set(ncList))
    return False

def makeFullSentence(sentence:list):
    sentence_string = ""
    for word in sentence:
        sentence_string += word.split("\t")[0].strip() + " "
    return sentence_string.strip()

def makeSentenceToken(sentence:list):
    # Use for NC labels + to get Verb SC's
    # so do TWO THINGS with this!
    nouns = getNCfromSentenceNouns(sentence)
    if nouns:
        # Only include sentences with nouns, else can't get SC reliably.
        verbs = getVerbs(sentence)
        # Below checks if there are verbs, and if so, returns sc_labels
        # If no sc_labels, also returns False
        sc_labels, badVerbs = labelAllSentenceVerbs(verbs, nouns)
        # Remove verbs that had no matching concords
        sentence = list(filter(lambda a: a not in badVerbs, sentence))
        

        if sc_labels: # sc_labels is False if no matching verbs for NCs
            # There have to noun and verb labels in this case
            labels = (makeNCLabels(nouns) +" "+" ".join(sc_labels)).strip()
            return (labels + " "+makeFullSentence(sentence))
        else:
            # If only nouns and no applicable verbs
            return (makeNCLabels(nouns) + " "+ makeFullSentence(sentence))  

def makeWordTokens(sentence:list):
    # Use for NC labels + to get Verb SC's
    # so do TWO THINGS with this!
    nouns = getNCfromSentenceNouns(sentence)
    labels = []
    
    # Only use sentences where a verb has a matching noun, or JUST noun. MUST HAVE NOUN(S)
    if nouns:
        # Only include sentences with nouns, else can't get SC reliably.
        verbs = getVerbs(sentence)
        labelled_verbs = getGoodVerbsWithLabel(verbs, nouns)
        if labelled_verbs:
            labels.extend(labelled_verbs)
        labels.extend(getLabelledNouns(sentence))
        return labels
    return False     

# tokenLevel: sentence or word
def getLabelledTokens(tokenLevel, out,inFile = IN_FILE):
    sentence = []
    with open(inFile, 'r', encoding='utf-8') as file, open(out, 'w', encoding='utf-8') as outfile:
        for line in file:
            line = line.strip()
            if not line: # If line is empty
                if len(sentence) > 0:
                    if tokenLevel == "sentence":
                        token = makeSentenceToken(sentence)
                        if token:
                            outfile.write(token+"\n")
                    elif tokenLevel == "word":
                        tokens = makeWordTokens(sentence)
                        if tokens:
                            for t in tokens:
                                outfile.write(t+"\n")
                    else:
                        print(f"Token level '{tokenLevel}' not recognised. Adjust and restart.")
                        return
                # write changes to file then reset below
                sentence = []
            else:
                # Only capture nouns and verbs, and of those, those that aren't "nonsense" (e.g. abbrevations)
                if notNoisy(line):
                    sentence.append(line.strip())

def main():
    choice = input("(1) Sentence, (2) word tokens, (3) both (4) full setences\n")
    match(choice):
        case("1"):
            getLabelledTokens("sentence", out = SENTENCE_FILE)
            generateTestTrain(0.80, SENTENCE_FILE, name="raw_sentence")
            print("Done")
        case("2"):
            getLabelledTokens("word")
            generateTestTrain(0.80, WORD_FILE, name="raw_word")
            print("Done")
        case("3"):
            getLabelledTokens("word", WORD_FILE)
            generateTestTrain(0.80, WORD_FILE, name="raw_word")
            print("Done with words")
            
            getLabelledTokens("sentence", SENTENCE_FILE)
            generateTestTrain(0.80, SENTENCE_FILE, name="raw_sentence")
            print("Done with sentences")
            print("Done")
        case("4"):
            print("Full sentences")
            getLabelledTokens("sentence", FULL_SENTENCE_FILE)
            generateTestTrain(0.80, FULL_SENTENCE_FILE, name="raw_full_sentence")
            print("Done with full sentences")
            print("Done")
