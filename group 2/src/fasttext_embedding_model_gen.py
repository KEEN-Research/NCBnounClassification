import fasttext

'''
Generate our skipgram embeddings based on the corpus
'''

file = './DataFiles/zu_clean_corpus.txt'

model = fasttext.train_unsupervised(file, model='skipgram', minn=2)

model.save_model('zu_fasttext_embeddings.bin')