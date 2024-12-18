# Cleans and formats data to my tagging protocol (see myTaggingProtocol.docx)
import re
import csv
import random
from collections import defaultdict

dataFile = ["dataA.txt", "dataB_1.txt","dataB_2.txt", "dataD.txt"]
DIRTY_FILE = "Data/dirty_data/zul_community_2017-sentences.txt"
CLEANED_RAWDATA = "Data/clean_data/rawCleaned.txt"
CANON_DATA = "Data/canonNouns/newCleanNouns.csv"
TRAINING_DATA = "Data/training/data.train"
TESTING_DATA = "Data/training/data.valid"
CLEAN_DATA = "Data/clean_data/cleanData.txt"

#* NB: When reformatting a labelled corpus, indicate which labels to keep/poss
#* by un/commenting labels here.
# Note: Data C is no longer was removed due to high overlap with another dataset.
tag_mappingABC = {
    "NPre": "NC",
    "BPre": "NC",
    "NPrePre": "NC",
    # "OC": "OC",
    # "SC": "SC",
    # "PossConc": "possC",
    # "Dem": "demC",
    # "AdjPref": "adjPre",
    # "VRoot": "verb",
    # "VTerm": "verb",
}

tag_mappingD = {
    "n": "NC",
    "iv_n": "NC",
    # "o": "OC",
    # "s": "SC",
    # "i": "SC",
    # "p": "SC",
    # # "z": "possC",
    # # "d": "demC",
    # "vr": "verb",
    # "vt": "verb"
}

def correctLabelsAB(labels:str, removeVerbs=True):
# format: ["__label__xxxx", "__label__xxxx"]
    if ("__label__verb" in labels): 
        if removeVerbs:
            labels = [label.replace('NC', 'SC') if label.startswith('__label__NC') else 
                  label for label in labels if label != "__label__verb"]
            return labels
        else:
            labels = [label.replace('NC', 'SC') if label.startswith('__label__NC') else 
                label for label in labels]
            return labels
    return labels
        
def formatA_to_B(string):
    """INPUT: String of format word\tx[tag]-y[tag]. Each input is line from source data.
    Returns: Word + ONLY concord/noun labels in lowercase
    """
    
    # for data A, where word/t tags
    if string.count("\t") ==1:
        word, tags= string.split("\t")
    # for data Bs, where word/t tags/t stem/t pos
    elif string.count("\t") ==3:
        word, tags, _,_ = string.split("\t")
     
    # removes duplicates if they arise   
    labels = set()
    
    # extract list of [tags]
    tags = re.findall(r'\[(.*?)\]', string)
    allLexItems = []
        
    for tag in tags:
        classNumber, lexicalItem = getItemAndNumber(tag)
        if lexicalItem != None:
            if lexicalItem in tag_mappingABC:
                # get matching dictionary tag + number from tag prefix
                labels.add("__label__"+tag_mappingABC[lexicalItem]+classNumber)
        else:
            tag = tag.strip()
            if tag in tag_mappingABC:
                # get matching dictionary tag + number from tag prefix
                labels.add("__label__"+tag_mappingABC[tag])

    if True:
        labels = correctLabelsAB(list(labels))

    # prepend labels, convert word to lowercase, return final line
    # remove in-line puncutation, e.g. . and % in 8.5%
    # if tabPos != -1 and list(labels) !=[]:-> removed bc removing morpheme method, which this is for
    if labels:
        word = re.sub(r"(\!|\%|\&|\"|\'|\?|\:|\-|\,|\.)", "", word)
        return (" ".join(labels) + " "+ word.lower())
    # if no applicable labels, return nothing
    return ""

def formatD(tagged_word):
    labels = set()
    
    # Get list of ALL tags in line (in tag part)
    found_tags = re.findall(r'\<(.*?)\>', tagged_word)
   
    # Go through all tags gotten
    for tag in found_tags:
        # Check if tag is in substituion list
        match tag:
            case("z3"):
                tagged_word = makeSubstituion(tagged_word, 'za<z3>', "ya")         
            case("iv_n1"|"iv_n3"|"iv_n11"|"iv_n14"):
            #sub char preceeding < with “u” (if it is O)
                tagged_word = makeSubstituion(tagged_word, 'o<'+tag+'>', "u")
            case("z4"):
                tagged_word = makeSubstituion(tagged_word, 'ka<z4>', "wa") 
            case("p1"):
                tagged_word = makeSubstituion(tagged_word,'e<p1>',"u" )
            case("p2"):
                tagged_word = makeSubstituion(tagged_word,'e<p2>',"ba" )
            case("p6"):
                tagged_word = makeSubstituion(tagged_word,'e<p6>',"a" )

        # for a <tagX>, get tag and X seperately
        classNum, lexicalItem = getItemAndNumber(tag)
        
        # check if tag is one we want a label mapping for
        if lexicalItem in tag_mappingD:
            
            # get matching dictionary tag + number from tag prefix
            
            final = "__label__"+tag_mappingD[lexicalItem]+classNum
            
            labels.add(final)

    if bool(labels):
        # remove all <tags> to get plain word, convert to lower and add labels
        word = re.sub(r'<[^>]*>', '', tagged_word)
        
        word = re.sub(r"(\!|\%|\&|\"|\'|\?|\:|\-|\,|\.)", "", word)
        
        return " ".join(list(labels)) + " "+ word.lower()
    
    return "" # if no applicable labels in word

def getItemAndNumber(tag):
    """get pos tag and class number from a tag
    Args:
        tag (string): any tag
    Returns:
        number (str): class number (first returned)
        item (str): pos item (second returned)
    """    
    # Assumes format: <itemXY>
    # Where < and > is any tag character e.g. []
    # item is any sequence of non-numeric characters
    # X = digit, Y = a or b or nothing
    
    match = re.search(r'\d+(a)?', tag)
    if match:
        # returns classNumber, lexicalItem (e.g. n, O, possConc, etc)
        return match.group(0), tag[0:match.start()]
    return None, None


    labels = set()
    
    #Get list of ALL tags in line (in tag part)
    found_tags = re.findall(r'\<(.*?)\>', tagged_word)
   
    # Go through all tags gotten
    for tag in found_tags:
        # Check if tag is in substituion list
        match tag:
            case("z3"):
                tagged_word = makeSubstituion(tagged_word, 'za<z3>', "ya")         
            case("iv_n1"|"iv_n3"|"iv_n11"|"iv_n14"):
            #sub char preceeding < with “u” (if it is O)
                tagged_word = makeSubstituion(tagged_word, 'o<'+tag+'>', "u")
            case("z4"):
                tagged_word = makeSubstituion(tagged_word, 'ka<z4>', "wa") 
            case("p1"):
                tagged_word = makeSubstituion(tagged_word,'e<p1>',"u" )
            case("p2"):
                tagged_word = makeSubstituion(tagged_word,'e<p2>',"ba" )
            case("p6"):
                tagged_word = makeSubstituion(tagged_word,'e<p6>',"a" )

        # for a <tagX>, get tag and X seperately
        classNum, lexicalItem = getItemAndNumber(tag)
        
        # check if tag is one we want a label mapping for
        if lexicalItem in tag_mappingD:
            
            # get matching dictionary tag + number from tag prefix
            
            final = "__label__"+tag_mappingD[lexicalItem]+classNum
            
            labels.add(final)

    if bool(labels):
        # remove all <tags> to get plain word, convert to lower and add labels
        word = re.sub(r'<[^>]*>', ' ', tagged_word)
        
        word = re.sub(r"(\!|\%|\&|\"|\'|\?|\:|\-|\,|\.)", "", word)
        
        return " ".join(list(labels)) + " "+ word.lower()
    
    return "" # if no applicable labels in word
    
def makeSubstituion(line, pattern, replacement):
    """Makes substituions where Ukwabelana tag does not 
        agree with actual prefix.

    Args:
        line (str): a tagged word
        pattern (str): what to look for, e.g. o<z4> (any <z4> tags preceeded by an o)
        replacement (str): what to replace ukwa substring with

    Returns:
        string: transformed string OR original if no conversions found
    """
    # find opening of tag in given pattern
    # build regexpression
    i = pattern.find("<")
    if (i==(-1)):
        i = pattern.find("[")
    pattern = pattern[:i] + "(?=" + pattern[i:] +")"
    
    # re.sub() returns NEW or original string if no replacements
    return re.sub(pattern, replacement, line)

def prepareSentenceData_AB(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
        sentence = []
        labels = set()
        for line in infile:
            if "<LINE" in line:
                continue

            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            elif len(parts) ==2: # for DataA
                
                word, tagged_word = parts 
            else:
                word, tagged_word,_,_ = parts 
            
            # # I.e. if this is a sentence ender
            # if "PUNC" in line:
            #     continue
            if word in [".", "?", "!", ";"]:
                print("end sentence")
                # Don't add in sentence enders -> cleaning
                #sentence.append(word)
                result = " ".join(sentence).lower() + "\n"
                result = re.sub(r",|'|\"|\(|\)", "", result)
                outfile.write(" ".join(list(labels)) +" "+result)
                
                # reset for new sentence
                sentence = []
                labels = set()
            else:
                sentence.append(word)
                labels.add(formatA_to_B(tagged_word))
        
        # Flush out a final sentence if exists
        if sentence:
            result = " ".join(sentence).lower() + "\n"
            outfile.write(" ".join(list(labels)) +" "+result)

def prepareData(inFile, setting="default", outFile=CLEAN_DATA, labels=True):
    """Accepts: file to clean/format 
    Writes all lines with nouns or concord agreements words with tags to cleanedData.txt
    Format: __tagX__ word
    Returns: NOTHING
    """
    if inFile == "Data/source/dataD.txt":
        with open(inFile, "r") as infile, open(outFile, "a") as outfile:
            for line in infile:
                if line.strip():
                    result = formatD(line)
                    if result.strip():
                        if labels:
                            # already has \n bc i forgot to strip it
                            outfile.write(result)
                        else:
                            outfile.write(result[result.rindex(" ")+1:]+"\n")
    else:
        with open(inFile, "r") as infile, open(outFile, "a") as outfile:
            for line in infile:
                if line[0].isalpha(): #only process word items
                    # remove unexpected tags not accounted for that cause label-less lines
                    result = formatA_to_B(line)
                    if result.strip():
                        if labels:
                            outfile.write(result+"\n")
                        else:
                            outfile.write(result[result.rindex(" ")+1:]+"\n")
                    
    print(f"Formatted and cleaned {inFile}")

def convertToCSV():    
    """Converts cleanedData.txt to .csv 
    """
    inputF = "Data/cleanData.txt"  
    output = "Data/cleanData.csv"
    
    with open(inputF, 'r') as txt_file, open(output, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row if needed
        csv_writer.writerow(['Labels', 'Word'])

        for line in txt_file:
            parts = line.strip().split() # label and word split
            
            # The last part is the word, the rest are labels
            word = parts[-1]
            labels = ' '.join(parts[:-1])
            
            csv_writer.writerow([labels, word])

    print("CSV file counterpart saved as: ", output)
   
def generateTestTrain(trainRatio:float=0.75, input_file='Data/clean_data/cleanData.txt', name="data"):

    # Define input and output file paths
    train_file = 'Data/training/'+name+'.train'
    test_file = 'Data/training/'+name+'.valid' 

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Shuffle the lines randomly 
    random.shuffle(lines)

    # Split index - going for 75/25 in most cases
    # Check if traiNRatio<50 to ensure it's train NOT test?
    splitIndex =  int(trainRatio * len(lines)) if (trainRatio>0.5) else int((1-trainRatio*len(lines)))
    
    # Split the data into training and testing sets
    train_data = lines[:splitIndex]
    test_data = lines[splitIndex:]

    # Write training data to train_file
    with open(train_file, 'w') as file:
        file.writelines(train_data)

    # Write testing data to test_file
    with open(test_file, 'w') as file:
        file.writelines(test_data)

    print(f"Data split into training: {len(train_data)} lines, testing: {len(test_data)} lines.")
    
def resetDataFile(data="Data/clean_data/cleanData.txt"):
    """Removes all content from cleanedData file
    """
    with open(data, "w") as file:
        file.write("")
        
    print("Reset data file.")

# CANON NOUN FILE FUNCTIONS---------------------------------------------------------

def fastTextifyCanonNouns(file=CANON_DATA, out="Data/canonNouns.valid"):
    with open(file, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        with open(out, mode='w', encoding='utf-8') as outfile:
            for row in csvreader: 
                noun = row[0]
                noun_class = row[1]
                outfile.write(f"__label__NC{noun_class} {noun.strip().lower()}\n")
    print(f"Converted file at {out} complete.")

def prepareSimplePOSData():
    file = "Data/source/SimplifiedPOS.ZU.cleanedData_Z.txt"
    out = "Data/clean_data/rawSimplePos.txt"
    print(f"Formatting {file}...")
    with open(file, "r") as infile, open(out, "w") as outfile:
        for line in infile:
            if line.strip() and ("\t" in line):
                w, pos = line.split("\t")
                if len(w.strip())>3: 
                    outfile.write(f"__label__{pos.strip()} {w.strip().lower()}\n")
    print(f"Done. Stored at {out}.")
    
def removeDuplicatesAndCompounds(infile):
    # NB: Sets don't retain adding order.
    unique_words = set()
    with open(infile, 'r') as file:
        for line in file:
            # Get the first word, so only first part of multiword words are included
            n, w = line.strip().split(maxsplit=1)
            
            unique_words.add(n+" "+w.split()[0].lower())
    
    # write changes to same file.
    with open(infile, 'w') as file:
        for word in unique_words:
            file.write(word + '\n')
    print(f"Processed file saved as {infile}")

# CORRECTLY FORMAT CRAWLED DATA ----------------------------------------------------
def preprocessDirtyData(text, language='z'):
    
    # Split the text into sentences based on ., ?, !, ..., or ;
    sentences = re.split(r'(?<=[.?!;])\s+', text)
    
    # Remove non-alpha characters at the start of each sentence
    cleaned_sentences = []
    for sentence in sentences:
        
        # Exclude lines with quotes as phrases etc as quote will detract from tagging (not SVO order)
        if  re.search(r'".*?"', sentence):
            continue
        cleaned_sentence = re.sub(r'^[^a-zA-Z]*', '', sentence)
        # Remove non-alpha word, non-whitespace characters (numbers, ounctuationetc)
        cleaned_sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        cleaned_sentence = cleaned_sentence.strip().lower()

         # Check if the sentence is not empty AND more than one word
        if (cleaned_sentence) and (cleaned_sentence.count(' ')>1):
                cleaned_sentences.append(cleaned_sentence)
                
    # Join the cleaned sentences with new lines so  sentence/line
    # Dont add where there were no full sentences
    if cleaned_sentences != []:
        cleaned_text = '\n'.join(cleaned_sentences) 
        return (cleaned_text+"\n")
    return None

def cleanDirtyData(rawData=DIRTY_FILE):
    
    resetDataFile(CLEANED_RAWDATA)
    with open(rawData, 'r', encoding='utf-8') as inFile, open (CLEANED_RAWDATA, 'w', encoding='utf-8') as outFile:
        for line in inFile:
            result = preprocessDirtyData(line)
            if result != None:
                outFile.write(result)
    # print(type(raw_data))
    # cleaned_data = preprocessDirtyData(raw_data)
    # with :
    #     outFile.write(cleaned_data)
    print(f"Data written tO {CLEANED_RAWDATA}")

# DATA REPORTING--- ----------------------------------------------------------------
def getDataReport(filename=TRAINING_DATA):
    pos_counts = defaultdict(int)
    class_counts = defaultdict(int)
    # Define a regex pattern to match labels
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                labels = (line.split(" ", maxsplit=1)[0]).replace("__label__", "")
                if labels == "v":
                    pos = "v"
                else:
                    classNum, pos= getItemAndNumber(labels)
                
                # Count the number of labels in the line
                pos_counts[pos] += 1
                class_counts[classNum] += 1
   
    pos_counts = dict(sorted(pos_counts.items(), key=lambda item: item[1]))
    class_counts =dict(sorted(class_counts.items(), key=lambda item: item[1]))
    print(f"POS Count Report for {filename}")
    for item, freq in pos_counts.items():
       print(item,": ", freq)
    print(f"Class Count Report for {filename}")
    for item, freq in class_counts.items():
       print(item,": ", freq)


# Script interaction------------------------------------------------------------------------

def testPlayground():
   getDataReport(TESTING_DATA)
   getDataReport(TRAINING_DATA)

fastTextifyCanonNouns(file="Data/canonNouns/alexNouns.csv",
                      out="Data/testing/alexNouns.txt")
def main():
    opt = input("""What'll it be tonight folks?\n 
                (3) Split prepared data into test/train\n
                (4) Fasttext-ify canon noun data\n
                (5) Formatting and splitting combo meal\n
                (6) Preprocess dirty raw data.\n
                (7) Make and split sentence data\n
                (8) Split cleaned raw sentence data""")   
    
    match(int(opt)):
        case 3:
            generateTestTrain(0.8)
        case 4:
            fastTextifyCanonNouns()
        case 5:
            classifierVerPath = "Data/clean_data/gold_nn_w_data.txt"
            classifierVerName = "gold_nn_w"
            resetDataFile(data = classifierVerName)
            for f in dataFile:
                prepareData("Data/source/"+f, outFile=classifierVerPath)
            generateTestTrain(0.8,input_file=classifierVerPath, name=classifierVerName)
        case 6:
            cleanDirtyData()
        case 7:
            sourceData = ["dataA.txt", "dataB_1.txt", "dataB_2.txt"]
            resetDataFile()
            for file in sourceData:
                inPath = "Data/source/"+file
                prepareSentenceData_AB(inPath,CLEAN_DATA)
                generateTestTrain(0.9)
        case 8:
            generateTestTrain(0.8, input_file=CLEANED_RAWDATA, name="raw")
        case _:
            print("Command not recognised.")

