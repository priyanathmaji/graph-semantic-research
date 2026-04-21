This dataset is based on the Roman Empire article from English Wikipedia, which
was selected since it is one of the longest articles on Wikipedia. The text was retrieved from the
English Wikipedia 2022.03.01 dump from Lhoest et al. (2021). 

https://huggingface.co/datasets/legacy-datasets/wikipedialegacy-datasets/wikipedia · 

Datasets at Hugging Face We’re on a journey to advance and democratize artificial intelligence through open source and open science.
 
Each node in the graph corresponds
to one (non-unique) word in the text. Thus, the number of nodes in the graph is equal to the article’s
length. 

Two words are connected with an edge if at least one of the following two conditions holds:
either these words follow each other in the text, 
or these words are connected in the dependency tree of the sentence (one word depends on the other). 

Thus, the graph is a chain graph with additional shortcut edges corresponding to syntactic dependencies between words. 

The class of a node is its syntactic role (we select the 17 most frequent roles as unique classes and group all the other roles
into the 18th class). 

The syntactic roles were obtained using spaCy (Honnibal et al., 2020). 
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The fast cat chases a mouse.")

    for token in doc:
        print(f"Token: {token.text}, Role: {token.dep_}, Head: {token.head.text}")

    # Output Snippet:
    # Token: cat, Role: nsubj, Head: chases
    # Token: chases, Role: ROOT, Head: chases
    # Token: mouse, Role: dobj, Head: chases

For node features, we use fastText word embeddings (Grave et al., 2018). 

While this task can probably be better solved with models from the field of NLP, we adapt it to evaluate GNNs in the setting of
low homophily, sparse connectivity, and potential long-range dependencies.
This graph has 22.7K nodes and 32.9K edges. 

By construction, the structure of this graph is chainlike; thus, it has the smallest average degree (2.9) and the largest diameter (6824). This graph is
heterophilous, hadj = −0.05. Interestingly, this dataset has a larger value of label informativeness
compared to all the other heterophilous datasets analyzed by Platonov et al. (2022). This means that
there are non-trivial label connectivity patterns specific to this dataset.
