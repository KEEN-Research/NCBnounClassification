import fasttext
import openpyxl

'''
Get the noun vectors
'''

model = fasttext.load_model('./ModelsAndVectors/zu_fasttext_embeddings.bin')

# Load the Excel file
workbook = openpyxl.load_workbook(f'./DataFiles/ZuluNounsSingleClass.xlsx')
worksheet = workbook.active

# Get all the class labels from the second column
nouns = [row[0].lower() for row in worksheet.iter_rows(min_row=1, values_only=True)]

noun_vectors = {}
for noun in nouns:
    noun_vectors[noun] = model.get_word_vector(noun)

# Save in FastText text format
with open('zu_noun_vectors.vec', 'w', encoding='utf-8') as f:
    f.write(f"{len(noun_vectors)} {model.get_dimension()}\n")
    for noun, vector in noun_vectors.items():
        vector_str = ' '.join(f"{v:.6f}" for v in vector) 
        f.write(f"{noun} {vector_str}\n")

