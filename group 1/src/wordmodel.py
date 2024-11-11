from gensim.models import FastText
import time
import fasttext
import matplotlib.pyplot as plt

FASTTEXT_MODEL = "Data/models/2024-04-26--fasttext-zu_monolingual-300-3.model"
RAW_DATA = "Data/source/cleanedData_Z.txt"

#! adjust minn / maxnn
def trainModel(filepath=RAW_DATA, minN=3, maxN=6, dimensions=150):
    print(f"Training new model: minN={minN}, maxN={maxN}, dimensions={dimensions}")
    model = fasttext.train_unsupervised(filepath, model="skipgram", minn=minN, maxn=maxN, dim=dimensions, epoch=15, ws=3)
    model.save_model(f"Data/models/zu_mine-{minN}_{maxN}_{dimensions}.bin")
    #model.save_model(f"Data/models/dumb.bin")

    print(f"Done! Getting nearest neighbours.")
    nn = model.get_nearest_neighbors("umfundisi",k=10)
    nn_p = []
    for n in nn:
        print(f"({n[1]}, {round(float(n[0]),3)})")
        nn_p.append(round(float(n[0]),3))
    print()
    return model




def experiment():
    dim = [150, 200,250,300]
    # for d in dim:
    #     trainModel(dimensions=d)
    results = [] # contains 8 sets of probabilitiees
    x = [1,2,3,4,5,6,7,8,9,10]
    
    minN = [2,3]
    maxN = [3,4,5,6]
    for mn in minN:
        for mx in maxN:
            results.append(trainModel(minN=mn, maxN=mx,dimensions=150))
    fig, ax = plt.subplots()
    
    resultSet= 0
    colours = ["darkgreen", "green", "lime","lightseagreen","darkorchid","thistle","mediumvioletred","fuchsia"]
    for mn in minN:
        for mx in maxN:
            ax.plot(x, results[resultSet], label=f"{mn}:{mx}",color=colours[resultSet])
            resultSet+=1
    baseline=  [(0.963), (0.938), (0.913), (0.907), (0.899), ( 0.898), (0.897), ( 0.897), 
                (0.895), (0.883)]
    ax.plot(x,baseline,  label="baseline",linewidth=1.5, color="grey")
    
    # Add a legend
    ax.legend()

    # Add titles and labels
    ax.set_title('Probability of Top-5 NN with Change in Character N-gram Window')
    ax.set_xlabel('NN closeness')
    ax.set_ylabel('Probability')

    plt.savefig("Data/results/ngram_window_wordmodel.png")
    # Show the plot
    plt.show()
    # In case show doesnt flush to the saved image
    plt.close()
    

def loadModel(filePath=FASTTEXT_MODEL,type="gensim"):
    print("Loading model...")
    load_t0  = time.time()
    if type == "gensim":
        vocab = FastText.load(FASTTEXT_MODEL)
        #vocab = FastTextKeyedVectors.load(FASTTEXT_MODEL, mmap='r')
    elif type == "facebook":
        vocab = FastText.load_fasttext_format("Data/models/zu_mine-3_6_150.bin")
    load_t1 = time.time()
    print(f"That took {(load_t1-load_t0)/60} minutes of your day!")
    print("Model loaded succesfully.")
    
    return vocab

def saveModel(newModelPath, model):
    print("Saving new model...")
    
    t0 = time.time()
    model.save(newModelPath)
    t1 = time.time()
    
    print(f"That took {(t1-t0)/60} minutes of your day!")
    print(f"Save succesful at {newModelPath}")
    
def getNearestNeighbours(model, word, quantity, type="classic"):
    # Returns [(word, probability)]
    
    match(type):
        case("classic"):
            nearest_neighbors = model.wv.most_similar(word.lower(),topn=quantity)
        case _:
            print("NN Type not recognised")
            return None
    #print(word,":", nearest_neighbors)
    return nearest_neighbors

def getVocabulary(model,name):
    # Open the output .vec file in write mode
    with open(f"Data/models/{name}.vec", 'w') as f_out:
        # Write the header: the number of words and the dimension of the vectors
        words = model.get_words()
        f_out.write(f"{len(words)} {model.get_dimension()}\n")
        
        # Write the word vectors
        for word in words:
            vector = model.get_word_vector(word)
            vector_str = ' '.join(map(str, vector))
            f_out.write(f"{word} {vector_str}\n")

def removeLabelsFromFile(file, outFile):
    print(f"Removing labels from {file}")
    with open(file, "r") as in_f, open(outFile, "w") as out_f:
        for line in in_f:
            out_f.write(line[line.rindex(" ")+1:])
