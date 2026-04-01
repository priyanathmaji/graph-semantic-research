This folder has the subroutines for the data preparation
1. Amazon Rating
   - Generate embedding using different libraries
   - Create the following files:
    A. CSV file with node features for each node (comma delimiter)
    B. Labels (for each node in line, provide the label, in case of multiple label, comma separated)
    C. Edges in simple txt file (a b) format where a and b are the node index. a, b -> 0....

Num (0-xxxx)  | Title | Popularity metric (salesrank) | Total Reviews | Category
Num (0-xxxx)  Average Rating
Num (0-xxxx) space Num (0-xxxx)   (min n=5 needed)

# Id of the book (Not used)
Id:   1
# Amazon Standard Identification Number (Node identifier)
ASIN: 0827229534
# Text of the title (Perform Semantic Embedding)
title: Patterns of Preaching: A Sermon Sampler

# Category (Perform 1 hot encoding - Book, DVD, Music CD, Videos)
group: Book

# Popularity metric (Numeric)
salesrank: 396585

# ASIN #s - Edges
# Drop the nodes that are not connected to 5 books (??)
similar: 5  0804215715  156101074X  0687023955  0687074231  082721619X

# Category of the books (4 levels of the taxonomy?)
categories: 2
|Books[283155]|Subjects[1000]|Religion & Spirituality[22]|Christianity[12290]|Clergy[12360]|Preaching[12368]
|Books[283155]|Subjects[1000]|Religion & Spirituality[22]|Christianity[12290]|Clergy[12360]|Sermons[12370]


# Total reviews
# Avg Rating - rating values into five classes (Prediction)
reviews: total: 2  downloaded: 2  avg rating: 5
# So, on July 28, 2000, customer A2JW67OY8U6HHK gave a 5-star rating. Out of 10 votes, 9 found the review helpful.
2000-7-28  cutomer: A2JW67OY8U6HHK  rating: 5  votes:  10  helpful:   9
2003-12-14  cutomer: A2VE83MZF98ITY  rating: 5  votes:   6  helpful:   5


Embedding Transformers
SBERT - pip install -U sentence-transformers