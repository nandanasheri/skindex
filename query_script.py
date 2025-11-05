import csv
from ranker import BM25, Ranker
from indexing import Indexer, IndexType
from document_preprocessor import RegexTokenizer
import json

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
    "cerave moisturizer for dry skin",
    "organic vegan free hair shampoo",
    "amla oil",
    "skincare for sun damage",
    "sunscreen for winter season",
    "conditionar cruelty free",
    "face wash for dry skin with pimples",
    "remove redness products",
    "organic body stuff",
    "cute sweaters for winter",
    "japanese sheet masks",
    "micellar water healthy skin",
    "remove dark circles under eyes",
    "brown skin foundation",
    "acne patches healthy",
    "natural deodarant tha doesn't smell bad",
    "frizzy hair curly hair products",
    "glycerin face serum for glowing skin",
    "lip balm that doesn't hurt my lips and is moistrurizing",
    "dandruff reducing shampoo that works for curly hair"
]


def map_docid_to_url(jsonl_path):
    """
    Reads a JSONL file and returns a dictionary mapping each docid to its product_url.
    
    Parameters:
        jsonl_path (str): Path to the JSONL file.
    
    Returns:
        dict: A mapping of {docid: product_url}.
    """
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # skip blank lines
            try:
                record = json.loads(line)
                docid = record.get("docid")
                url = record.get("product_url")
                if docid is not None and url:
                    mapping[docid] = url
            except json.JSONDecodeError:
                print("Skipping invalid JSON line:", line)
    
    return mapping

docid_to_url = map_docid_to_url('./data/inci_products.jsonl')

# write to output file
with open("groundtruth.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query", "product_id", "url"])
    for query in queries:
        # rank each query against corpus & assign relevance score based on rank
        relevance_score = ranker.query(query)
        for i, (docid, score) in enumerate(relevance_score[:50]):
            writer.writerow([query, docid, docid_to_url[docid]])