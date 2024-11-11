import openpyxl

'''
functionality for labelling our courpus using rule based approach, corpus pre labbelled for noun information
'''

prefixingInformation = {
    '1a':{
        'scC': 'u',
        'scAE':'w',
        'scO': 'w',
        'scNeg': 'ka',
        'scCon': 'e',
        'scSubj': 'a'
    },
    '1':{
        'scC': 'u',
        'scAE':'w',
        'scO': 'w',
        'scNeg': 'ka',
        'scCon': 'e',
        'scSubj': 'a'
    },
    '2':{
        'scC': 'ba',
        'scAE':'b',
        'scO': 'b',
        'scNeg': '+',
        'scCon': 'be',
        'scSubj': '+'
    },
    '2a':{
        'scC': 'ba',
        'scAE':'b',
        'scO': 'b',
        'scNeg': '+',
        'scCon': 'be',
        'scSubj': '+'
    },
    '3':{
        'scC': 'u',
        'scAE':'w',
        'scO': 'w',
        'scNeg': 'wu',
        'scCon': '+',
        'scSubj': '+'
    },
    '4':{
        'scC': 'i',
        'scAE':'y',
        'scO': 'y',
        'scNeg': 'yi',
        'scCon': '+',
        'scSubj': '+'
    },
    '5':{
        'scC': 'li',
        'scAE':'l',
        'scO': 'l',
        'scNeg': '+',
        'scCon': '+',
        'scSubj': '+'
    },
    '6':{
        'scC': 'a',
        'scAE':'',
        'scO': '',
        'scNeg': 'wa',
        'scCon': 'e',
        'scSubj': '+'
    },
    '7':{
        'scC': 'si',
        'scAE':'s',
        'scO': 's',
        'scNeg': '+',
        'scCon': '+',
        'scSubj': '+'
    },
    '8':{
        'scC': 'zi',
        'scAE':'z',
        'scO': 'z',
        'scNeg': '+',
        'scCon': '+',
        'scSubj': '+'
    },
    '9':{
        'scC': 'i',
        'scAE':'y',
        'scO': 'y',
        'scNeg': 'yi',
        'scCon': '+',
        'scSubj': '+'
    },
    '10':{
        'scC': 'zi',
        'scAE':'z',
        'scO': 'z',
        'scNeg': '+',
        'scCon': '+',
        'scSubj': '+'
    },
    '11':{
        'scC': 'lu',
        'scAE':'lw',
        'scO': 'l',
        'scNeg': '+',
        'scCon': '+',
        'scSubj': '+'
    },
    '14':{
        'scC': 'bu',
        'scAE':'b',
        'scO': 'b',
        'scNeg': '+',
        'scCon': '+',
        'scSubj': '+'
    },
    '15':{
        'scC': 'ku',
        'scAE':'kw',
        'scO': 'k',
        'scNeg': '+',
        'scCon': '+',
        'scSubj': '+'
    },
}

#function to check if item is consonant
def isConsonant(x): 
    if (x == 'a' or x == 'e' or x == 'i' or 
        x == 'o' or x == 'u' or x == 'A' or 
        x == 'E' or x == 'I' or x == 'O' or 
        x == 'U'): 
        return False 
    else: 
        return True 


# Load the Excel file
workbook = openpyxl.load_workbook(f'./DataFiles/ZuluNounsSingleClass.xlsx')
worksheet = workbook.active


with open('./DataFiles/zu_verb_roots.txt', 'r', encoding='utf-8') as input_file:
    verbs = input_file.readlines()
    verbs = [s.strip() for s in verbs]


# Get all the class labels from the second column
nouns = [(row[0].lower(),row[1]) for row in worksheet.iter_rows(min_row=1, values_only=True)]


def clean_and_save_text_for_fasttext(input_file_path, output_file_path):
    
    # Open and read the input file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    # Define a function to clean each line
    def clean_line(line):
        global number_of_lines
        global progress 
        words = line.split(" ")
        words = [s.strip() for s in words]
        label_str = ""

        last_noun=None

        #apply labels based on rule-based logic, following a Subject verb object format that is apparent in Bantu languages
        for word in words:
            noun_match = [(w,s) for w,s in nouns if w==word]
            if noun_match:
                label_str += "__label__"+str(noun_match[0][1])+" "
                last_noun = str(noun_match[0][1])
            else:
                verb_match = [v for v in verbs if v in word and not word.startswith(v) and last_noun]
                if verb_match:
                    root = max(verb_match, key = len)
                    front = word[0:word.index(root)+1]
                    
                    subjectConcords = (prefixingInformation[last_noun])  
                                        
                    if front.startswith("a") and last_noun not in ['1','1a','6']:
                   
                        if(subjectConcords['scNeg']) == '+':
                            
                            front=front[1:]

                            char_after = front[subjectConcords['scC'].__len__():subjectConcords['scC'].__len__()+1]
                            if(front.startswith(subjectConcords['scC']) and isConsonant(char_after) ):
                                label_str += "__label__SC"+last_noun+" "
                                continue

                            char_after = front[subjectConcords['scAE'].__len__():subjectConcords['scAE'].__len__()+1]
                            if(front.startswith(subjectConcords['scAE']) and char_after=="a" or char_after=="e"):
                                label_str += "__label__SC"+last_noun+" "
                                continue

                            char_after = front[subjectConcords['scO'].__len__():subjectConcords['scO'].__len__()+1]
                            if(front.startswith(subjectConcords['scO']) and char_after=="o"):
                                label_str += "__label__SC"+last_noun+" "
                                continue

                        else:
                            
                            if(front[1:].startswith(subjectConcords['scNeg'])):
                                label_str += "__label__SC"+last_noun+" "
                                continue
                    else:
                        
                        char_after = front[subjectConcords['scC'].__len__():subjectConcords['scC'].__len__()+1]
                        if(front.startswith(subjectConcords['scC']) and isConsonant(char_after) ):
                            label_str += "__label__SC"+last_noun+" "
                            continue

                        char_after = front[subjectConcords['scAE'].__len__():subjectConcords['scAE'].__len__()+1]
                        if(front.startswith(subjectConcords['scAE']) and char_after=="a" or char_after=="e"):
                            label_str += "__label__SC"+last_noun+" "
                            continue

                        char_after = front[subjectConcords['scO'].__len__():subjectConcords['scO'].__len__()+1]
                        if(front.startswith(subjectConcords['scO']) and char_after=="o"):
                            label_str += "__label__SC"+last_noun+" "
                            continue

                        if(front.startswith(subjectConcords['scCon'])):
                            label_str += "__label__SC"+last_noun+" "
                            continue

                        if(front.startswith(subjectConcords['scSubj'])):
                            label_str += "__label__SC"+last_noun+" "
                            continue
        progress+=1
        print(str(progress) +' out of ' + str(number_of_lines), end='\r')
        return label_str + " ".join(words)

    global number_of_lines
    number_of_lines =lines.__len__()
    global progress
    progress=0

    # Apply the cleaning function to each line
    cleaned_lines = [clean_line(line) for line in lines]

    # Write the cleaned lines to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for cleaned_line in cleaned_lines:
            output_file.write(cleaned_line + '\n')


clean_and_save_text_for_fasttext("./DataFiles/zu_clean_corpus.txt","zu_labelled_corpus.txt")