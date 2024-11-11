from annoy import AnnoyIndex
from collections import Counter


#function to get the most frequent item in list
def most_frequent(lst):
    if not lst:
        return None
    
    # Count occurrences of each item
    count = Counter(lst)
    
    # Find the maximum frequency
    max_count = max(count.values())
    
    # Get all items with the maximum frequency
    most_frequent_items = [item for item, freq in count.items() if freq == max_count]
    
    # Return the alphabetically first item among the most frequent
    return min(most_frequent_items)

#function check if item in unique list
def is_unique(unique_prefixes, query):
    for (value,prefix) in unique_prefixes:
        if query.startswith(prefix):
            return value
    return None

#function get unique prefixes out of prefix list
def get_unique_prefixes(prefixDict):
    uniquePrefix = []
    for (value,prefix) in prefixDict:
        isUnique = True
        for value2, otherPrefix in prefixDict:

            if(len(otherPrefix) >= len(prefix) and (value != value2)):
                if otherPrefix.startswith(prefix):
                    isUnique = False
                    break
        if isUnique:
            uniquePrefix.append((value,prefix))

    return uniquePrefix

#function get semantic annoy neighbours
def get_semantic_noun_neigbours(nouns, semantic_model,indexer, query_word, topn):
    
    nearest = semantic_model.wv.most_similar(query_word, topn=1000, indexer=indexer)
    
    noun_neighbors = [(w) for (w,x) in nearest if (w in nouns)][1:topn+1]
    
    return noun_neighbors