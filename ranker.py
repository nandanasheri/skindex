"""
Rankers - Specifically BM25
"""
from collections import Counter, defaultdict
import numpy as np
from indexing import InvertedIndex
import random


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, product_index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', ingredient_index: InvertedIndex = None, raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
        """
        self.product_index = product_index
        self.ingredient_index = ingredient_index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query (Hint: Also apply stopwords filtering to the tokenized query)
        tokens = self.tokenize(query)
        # Filter out None/stopwords completely for the Counter
        valid_tokens = [t for t in tokens if t not in self.stopwords and t is not None]

        query_word_counts = Counter(valid_tokens)
        
        # 2. Build the candidate dictionary
        # Structure: {docid: {term: {'prod': freq, 'ing': freq}}}
        doc_data = defaultdict(lambda: defaultdict(lambda: {'prod': 0, 'ing': 0}))

        for token in set(valid_tokens):
            # A. Product Index Postings
            if token in self.product_index.vocabulary:
                product_postings = self.product_index.get_postings(token)
                for docid, freq in product_postings:
                    # Store count of tokens in each doc in product index
                    doc_data[docid][token]['prod'] = freq
            if self.ingredient_index:
                # B. Ingredient Index Postings
                if token in self.ingredient_index.vocabulary:
                    ingredient_postings = self.ingredient_index.get_postings(token)
                    for docid, freq in ingredient_postings:
                        # Store count of tokens in each doc in ingredient index
                        doc_data[docid][token]['ing'] = freq
        
        # 3. Score Documents
        all_scores = []
        for doc_id, term_counts in doc_data.items():
            # term_counts is: {'acne': {'prod': 1, 'ing': 0}, 'cream': {'prod': 5, 'ing': 0}}
            score = self.scorer.score(doc_id, term_counts, query_word_counts)
            all_scores.append((doc_id,score))

        # 2.1 For each token in the tokenized query, find out all documents that contain it and counting its frequency within each document.
        # Hint 1: To understand why we need the info above, pay attention to docid and doc_word_counts, 
        #    located in the score() function within the RelevanceScorer class
        # Hint 2: defaultdict(Counter) works well in this case, where we store {docids : {query_tokens : counts}}, 
        #         or you may choose other approaches

        # 2.2 Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes) 
        #        for each relevant document determined in 2.1

        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)

        return sorted_scores


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # NOTE: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict, query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document that are also found in the query, 
                                and their frequencies within the document.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError
    
# Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        avdl = self.index.get_statistics()['mean_document_length']
        num_docs = self.index.get_statistics()['number_of_documents']
        doc_len = self.index.get_doc_metadata(docid)['length']

        b = self.b
        k1 = self.k1
        k3 = self.k3
        score = 0
        # 2. Compute additional terms to use in algorithm
        for q_term in query_word_counts:
            # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
            if q_term and q_term in self.index.index:
                doc_tf = doc_word_counts[q_term]['prod'] # term frequency
                if doc_tf > 0:
                    # document frequency
                    df = self.index.get_term_metadata(q_term)["doc_frequency"]
                    idf = np.log((num_docs - df + 0.5) / float(df + 0.5))
                    qtf = query_word_counts[q_term]
                    tf = ((k1 + 1) * doc_tf) / float((k1 * (1 - b + (b * (doc_len/float(avdl))))) + doc_tf)
                    qtf_norm = ((k3 + 1) * qtf) / float(k3 + qtf)
                    bm25 = qtf_norm * tf * idf
                    score += bm25
        
        return score
    
class BM25F(RelevanceScorer):
    def __init__(self, product_index: InvertedIndex, ingredient_index: InvertedIndex, parameters: dict = {'b': 0.1, 'k1': 1.2, 'k3': 8, 'w_prod': 5.0, 'w_ing': 1.0}) -> None:
        self.product_index = product_index
        self.ingredient_index = ingredient_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.w_prod = parameters['w_prod']
        self.w_ing = parameters['w_ing']
        
        self.stats_prod = self.product_index.get_statistics()
        self.stats_ing = self.ingredient_index.get_statistics()

    def score(self, docid: int, doc_word_counts: dict[str, dict[str, int]], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from each index
        # Product Index (Product Description)
        try:
            prod_doc_len = self.product_index.get_doc_metadata(docid)['length']
        except KeyError:
            prod_doc_len = 0
        avdl_prod = self.stats_prod['mean_document_length']

        # Ingredient Index (Ingredient Function)
        try:
            ingredient_doc_len = self.ingredient_index.get_doc_metadata(docid)['length']
        except KeyError:
            ingredient_doc_len = 0
        avdl_ing = self.stats_ing['mean_document_length']

        score = 0.0

        # 2. Iterate through query terms
        for q_term, qtf in query_word_counts.items():
            # Reset TFs for every term
            tf_prod = 0.0
            tf_ing = 0.0
            
            # 3. Retrieve TF specific to each field
            term_data = doc_word_counts.get(q_term, {'prod': 0, 'ing': 0})
            tf_prod = term_data['prod']
            tf_ing = term_data['ing']
                        
            if tf_prod == 0 and tf_ing == 0:
                continue
            
            # 4. Normalize Frequencies (The "F" part of BM25F)
            # Calculate denominator for Product Description
            denom_prod = 1.0
            if avdl_prod > 0:
                denom_prod = 1 - self.b + self.b * (prod_doc_len / avdl_prod)
                
            # Calculate denominator for Ingredient Function
            denom_ing = 1.0
            if avdl_ing > 0:
                denom_ing = 1 - self.b + self.b * (ingredient_doc_len / avdl_ing)

            # Combine the weighted, normalized frequencies
            w_tf = (self.w_prod * tf_prod / denom_prod) + (self.w_ing * tf_ing / denom_ing)
            
            # 5. Apply Saturation (The BM25 part)
            saturation = w_tf / (self.k1 + w_tf)
            
            # 6. Calculate IDF
            # Document frequency for each index
            df_product = 0
            if q_term in self.product_index.vocabulary:
                df_product = self.product_index.get_term_metadata(q_term)["doc_frequency"]
            df_ingredient = 0
            if q_term in self.ingredient_index.vocabulary:
                df_ingredient = self.ingredient_index.get_term_metadata(q_term)["doc_frequency"]
           
            df = max(df_product, df_ingredient)
            
            # Get N (Total docs) safely
            num_docs_prod = self.stats_prod.get('number_of_documents', 0)
            num_docs_ing = self.stats_ing.get('number_of_documents', 0)
            N = max(num_docs_prod, num_docs_ing)
            
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            # 7. Query Term Weighting (k3)
            qtf_norm = ((self.k3 + 1) * qtf) / (self.k3 + qtf)
            
            # 8. Final Score
            bm25 = qtf_norm * saturation * idf
            score += bm25

        return score

class RandomScorer(RelevanceScorer):
    def __init__(self, index, parameters=None):
        self.index = index

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        return random.random()

class DocLengthScorer(RelevanceScorer):
    def __init__(self, index, parameters=None):
        self.index = index

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        return float(self.index.get_doc_metadata(docid)['length'])
