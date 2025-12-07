"""
Rankers - Specifically BM25
"""
from collections import Counter, defaultdict
import numpy as np
from indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, product_index: InvertedIndex, ingredient_index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
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
        valid_tokens = []

        for token in tokens:
            if token not in self.stopwords:
                valid_tokens.append(token)
            else:
                valid_tokens.append(None)

        query_word_counts = Counter(valid_tokens)
        doc_word_counter = defaultdict(Counter)

        for token in valid_tokens:
            docs_w_token = self.product_index.index[token]
            if not token:
                continue
            for doc in docs_w_token:
                doc_word_counter[doc[0]][token] = doc[1]
            
            docs_w_token_ingred = self.ingredient_index.index[token]
            if not token:
                continue
            for doc in docs_w_token_ingred:
                doc_word_counter[doc[0]][token] += doc[1]
        
        all_scores = []
        for doc_id in doc_word_counter:
            score = self.scorer.score(doc_id, doc_word_counter[doc_id], query_word_counts)
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

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
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
                doc_tf = doc_word_counts[q_term] # term frequency
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
    def __init__(self, product_index: InvertedIndex, ingredient_index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8, 'w_prod': 1.0, 'w_ing': 2.0}) -> None:
        self.product_index = product_index
        self.ingredient_index = ingredient_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.w_prod = parameters['w_prod']
        self.w_ing = parameters['w_ing']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from each index

        # Product Index (Product Description)
        prod_desc_avdl = self.product_index.get_statistics()['mean_document_length']
        prod_desc_num_docs = self.product_index.get_statistics()['number_of_documents']
        try:
            prod_desc_doc_len = self.product_index.get_doc_metadata(docid)['length']
        except KeyError:
            prod_desc_doc_len = 0

        # Ingredient Index (Ingredient Function)
        ingredient_avdl = self.ingredient_index.get_statistics()['mean_document_length']
        ingredient_num_docs = self.ingredient_index.get_statistics()['number_of_documents']
        try:
            ingredient_doc_len = self.ingredient_index.get_doc_metadata(docid)['length']
        except KeyError:
            ingredient_doc_len = 0

        # Hyperparameters
        b = self.b
        k1 = self.k1
        k3 = self.k3
        score = 0.0

        # 2. Compute additional terms to use in algorithm
        for q_term in query_word_counts:
            tf_prod = 0.0
            tf_ingred = 0.0
            # 3. Compute TF for product index
            if q_term and q_term in self.product_index.index:
                prod_doc_tf = doc_word_counts[q_term] # term frequency
                if prod_doc_tf > 0:
                    tf_prod = ((k1 + 1) * prod_doc_tf) / float((k1 * (1 - b + (b * (prod_desc_doc_len/float(prod_desc_avdl))))) + prod_doc_tf)
            # 4. Compute TF for ingredient index
            if q_term and q_term in self.ingredient_index.index:
                ingred_doc_tf = doc_word_counts[q_term] # term frequency
                if ingred_doc_tf > 0:
                    tf_ingred = ((k1 + 1) * ingred_doc_tf) / float((k1 * (1 - b + (b * (ingredient_doc_len/float(ingredient_avdl))))) + ingred_doc_tf)

            # 5. Get total tf
            tf_doc = self.w_prod * tf_prod + self.w_ing * tf_ingred

            # document frequency for each index
            df_product = self.product_index.get_term_metadata(q_term)["doc_frequency"]
            df_ingredient = self.ingredient_index.get_term_metadata(q_term)["doc_frequency"]
            df = df_product + df_ingredient
            total_docs = max(prod_desc_num_docs, ingredient_num_docs)
            idf = np.log((total_docs - df + 0.5) / (df + 0.5))
            # compute idf value based on max(doc_freq)
            # blended_idf = max(np.log((prod_desc_num_docs - df_product + 0.5) / float(df_product + 0.5)), (np.log((ingredient_num_docs - df_ingredient + 0.5) / float(df_ingredient + 0.5))))
            qtf = query_word_counts[q_term]
            # qtf_norm = ((k3 + 1) * qtf) / float(k3 + qtf)
            qtf_norm = (((k3 + 1) * qtf) / (k3 * qtf))
            bm25 = qtf_norm * tf_doc * idf
            score += bm25

        return score