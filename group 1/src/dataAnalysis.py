import csv
from collections import defaultdict
import re
from wordmodel import getNearestNeighbours, loadModel

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


#best =[ "uthi", "abantu", "into", "izwe", "umuntu"]
best = ["into"]
middle = ["iqola"]
worst = ["imithuqasi"]

#a', 106), ('amaqola', 106), ('iqola', 105)], worst: [('imithuqasi', 1), ('isixhumanisi', 1), ('usenzenjani', 1), ('izintekenteke', 1), ('isenyelo', 1)]
# Print or save the frequencies
# for noun, freq in sorted_nouns.items():
#     print(f'{noun}: {freq}')
