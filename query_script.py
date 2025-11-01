import csv
from ranker import BM25, Ranker
from indexing import Indexer, IndexType
from document_preprocessor import RegexTokenizer

# initialize stopwords
stopwords = set()
with open('data/stopwords.txt', 'r') as f:
    for line in f:
        stopwords.add(line.strip())
f.close()

# intialize preprocesser, index, and ranker
preprocessor = RegexTokenizer()
index = Indexer.create_index(
    IndexType.BasicInvertedIndex, './data/inci_products.jsonl', preprocessor, stopwords, 1, "desc")
scorer = BM25(index)
ranker = Ranker(index, preprocessor, stopwords, scorer)

# just for viewing index
# with open("index.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["keyword", "product_id"])
#     for k, v in index.index.items():
#         writer.writerow([k, v])

# list of 20 queries 
queries = [
    "hydrating cleanser",
    "gentle exfoliating cleaner",
    "vitamin c sunscreen",
    "adapalene gel that doesn't burn my skin",
    "sunscreen for acne-prone skin",
    "cleaner that doesn't strip away my natural oils",
    "toner that maintains my skin barrier",
    "products that treat Eczema",
    "vegan skincare",
    "moisturizer for humid climates",
    "korean skincare",
    "makeup removal products",
    "best sunscreen for UV protection",
    "how to repair damaged skin barrier",
    "spot treatment for acne",
    "most popular anti-oxidant in united states during the summer",
    "clinically recommended chemical peel",
    "products that can treat closed comedones fast",
    "moisturizer that has vitamin e in it",
    "oil-based cleanser to remove makeup"
]

# write to output file
with open("baseline_relevance_score.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query", "product_id", "relevance_score"])
    for query in queries:
        # rank each query against corpus & assign relevance score based on rank
        relevance_score = ranker.query(query)
        for i, (docid, score) in enumerate(relevance_score[:50]):
            if i < 10: # top 10 docs
                relevance = 5
            elif i < 20: # next 10
                relevance = 4
            elif i < 30: # next 10
                relevance = 3
            elif i < 40: # next 10
                relevance = 2
            else:
                relevance = 1
            writer.writerow([query, docid, relevance])