## Skindex

### Components within the Directory: 

- build index using  `indexing.py` and  `document_preprocessor.py`
- rank documents using  BM25FRanker under `ranker.py`
- run relevance tests for MAP and NDCG using  `relevance.py` 
- all scraping logic can be found under  `scrape.ipynb` 
- full IR system logic can be found implemented under  `skindex.ipynb` 

### How to run full Skindex Pipeline
- from indexing to ranking and relevance, each step is documented within the  `skindex.ipynb`
- ensure `data/` has the following files : 
    -  ` 'inci_products_new.jsonl'`  
    -  ` 'inci_ingredient_functions.jsonl' ` 
    -  ` 'groundtruth.csv' ` 
    -  ` 'relevance_results.json' ` 