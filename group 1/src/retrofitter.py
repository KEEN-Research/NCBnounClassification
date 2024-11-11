import xml.etree.ElementTree as ET
from collections import defaultdict
import fasttext
from alive_progress import alive_bar

def have_overlapping_prefix(word1, synonym):
    # word/synonym distinction is arbitrary here, just for clarity
    
    # Find the minimum length of the two words
    min_length = min(len(word1), len(synonym))
    
    # Iterate through the prefixes
    for i in range(1, min_length + 1):
        if word1[:i] == synonym[:i]:
            return True
    return False

def parse_synset_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Create a dictionary to store synsets
    lexicon = defaultdict(list)

    # Find the Lexicon element (the parent of LexicalEntry elements)
    lexicon_element = root.find('Lexicon')

    # Loop through each LexicalEntry within the Lexicon
    for entry in lexicon_element.findall('LexicalEntry'):
        lemma = entry.find('Lemma').attrib['writtenForm']
        pos = entry.find('Lemma').attrib['partOfSpeech']

        # Only process nouns
        if pos == 'n':
            synsets = set()  # To avoid duplicates
            for sense in entry.findall('Sense'):
                synset = sense.attrib['synset']
                synsets.add(synset)

            # Add each noun to the lexicon based on shared synsets
            for synset in synsets:
                for entry2 in lexicon_element.findall(f".//Sense[@synset='{synset}']/.."):
                    related_lemma = entry2.find('Lemma').attrib['writtenForm']
                    if related_lemma != lemma:
                        #* Crude "nc-verifying" step, to help include only synonyms of same class
                        if have_overlapping_prefix(lemma, related_lemma):
                            lexicon[lemma].append(related_lemma)

    # Convert defaultdict to a regular dict
    return dict(lexicon)

def write_lemmas_to_file(lexicon, output_file):
    with open(output_file, 'w') as f:
        for lemma, related_lemmas in lexicon.items():
            related_lemmas_str = ' '.join(related_lemmas)
            f.write(f"{lemma} {related_lemmas_str}\n")
            
import fasttext
from gensim.models import FastText

def convert_model_to_vec(input_model_path, output_vec_path):
    print("Converting to .vec")
    #* FastText native
    if input_model_path.endswith('.bin'):
        # Load the FastText model from the .bin file (native fastText format)
        model = fasttext.load_model(input_model_path)
        words = model.get_words()
        dim = model.get_dimension()
        
        with open(output_vec_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"{len(words)} {dim}\n")
            
            for word in words:
                vector = model.get_word_vector(word)
                vector_str = ' '.join(map(str, vector))
                f_out.write(f"{word} {vector_str}\n")
    #* Gensim
    elif input_model_path.endswith('.model'):
        # Load the FastText model trained with Gensim
        print("Loading model...")
        model = FastText.load(input_model_path)
        words = model.wv.index_to_key
        dim = model.vector_size
        print("Done! Getting vectors...")
        with open(output_vec_path, 'w', encoding='utf-8') as f_out:
            # Write the header line: number of words and dimensions
            size= len(words)
            f_out.write(f"{size} {dim}\n")
            
            # Write each word and its corresponding vector
            with alive_bar(size) as bar:
                for word in words:
                    vector =  model.wv[word]
                    vector_str = ' '.join(map(str, vector))
                    f_out.write(f"{word} {vector_str}\n")
                    bar()
    else:
        raise ValueError("Unsupported file format. Please provide a .bin or .model file.")
    print(f"Done, saved {input_model_path} to {output_vec_path}")




# input_model_file = "Data/models/zu_mine-2_6_150.bin" # or 'path/to/your/model.model'
# output_vec_file = "Data/retrofitting/mine_2_6_vectors.vec"
# convert_model_to_vec(input_model_file, output_vec_file)

# # Example usage
input_model_file = "Data/models/2024-04-26--fasttext-zu_monolingual-300-3.model" # or 'path/to/your/model.model'
output_vec_file = "Data/retrofitting/INCvectors.vec"
convert_model_to_vec(input_model_file, output_vec_file)