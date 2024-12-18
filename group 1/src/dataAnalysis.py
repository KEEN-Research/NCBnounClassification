from collections import defaultdict, Counter
import re
# from main import extractTagAndClass


def extractTagAndClass(word):
    """Returns tag and class from label of form XXXNN
    Args:
        word (string): label or label and word with tag/class
    Returns:
        tag (String): pos/morpheme label
        class_part (String): class number
        OR
        None, None
        
    """
    # Use regular expression to separate the tag and class
    match = re.match(r"([a-zA-Z]+)(\d+[a|b]?)", word)
    if match:
        tag = match.group(1)
        class_part = match.group(2)
        return tag, class_part
    else:
        return None, None  # Return None if the word doesn't match the expected format

def count_label_frequencies(file_path):
    # Initialize a counter to store the frequencies
    label_counts = Counter()
    
    # Use a regex pattern that matches '__label__' followed by any characters and captures the ending number
    pattern = re.compile(r'__label__\S*?(\d+[a]?)\b')

    with open(file_path, 'r') as file:
        for line in file:
            
            matches = pattern.findall(line)
            for match in matches:
                label_counts[match] += 1

    return list(label_counts.items())

def getMostFrequentVocabulary():
    # Load the wordlist CSV
    wordlist_file = 'Data/source/INC_wordlist.txt'
    noun_list_file = 'Data/canonNouns/canonTest_noNC.txt' # No labels


    # Read the list of nouns
    with open(noun_list_file, 'r', encoding='utf-8') as file:
        nouns = [line.strip() for line in file]

    # Initialize a dictionary to hold frequencies
    noun_frequencies = defaultdict(int)

    print("Processing frequencies...")
    # Process the wordlist
    with open(wordlist_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Split by multiple tabs using regular expressions
            row = re.split(r'\t+', line.strip())
            if len(row) == 3:  # Ensure correct format
                _, word, frequency = row
                frequency = int(frequency)
                
                # Check if any noun is a substring of the word
                for noun in nouns:
                    if noun.lower() in word:
                        noun_frequencies[noun.lower()] += frequency
    print("Done!")


    sorted_nouns = (sorted(noun_frequencies.items(), key=lambda item: item[1], reverse=True))
    top3 = sorted_nouns[:5]
    midN =len(sorted_nouns)//2
    mid = sorted_nouns[midN:midN+5]
    worst = sorted_nouns[-5:]

    print(f"Top3: {top3}, middlemost: {mid}, worst: {worst}")

def getClassCount(file):
    items = []
    with open(file, "r") as testFile:
        for line in testFile:
            item = line.strip().split(" ", maxsplit =1)[0].split("__label__")[1]
            t, nc = extractTagAndClass(item) 
            items.append(str(nc))
    
    counts = dict(Counter(items))

    with open("Data/results/keet_nc_counts.csv", "w") as resultFile:
        resultFile.write("class;%;count\n")
        for item in sorted(list(counts)):
            
            # write class
            resultFile.write(str(item)+";")
            
            # write %
            resultFile.write(str(round((counts[item]/724) *100 , 2)))
            
            # write count
            resultFile.write(";"+str(counts[item])+"\n")
        
    print(counts)
            
