import fasttext

'''
Train a fasstext classifer using labeled corpus
'''

model = fasttext.train_supervised(input="./DataFiles/zu_labelled_corpus.txt",pretrainedVectors='./ModelsAndVectors/zu_noun_vectors.vec', minn=2)
model.save_model("zu_fasttext_classifer.bin")

    