import numpy as np
from collections import Counter, defaultdict
import numpy as np
from indexing import InvertedIndex
import random
from ranker import RelevanceScorer

class BM25F(RelevanceScorer):
    def __init__(self, product_index: InvertedIndex, ingredient_index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8, 'w_prod': 1.0, 'w_ing': 1.0}) -> None:
        self.product_index = product_index
        self.ingredient_index = ingredient_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.w_prod = parameters['w_prod']
        self.w_ing = parameters['w_ing']
        
        # Pre-fetch stats to improve performance
        self.stats_prod = self.product_index.get_statistics()
        self.stats_ing = self.ingredient_index.get_statistics()

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get Document Lengths (Safely)
        # We use try-except because a docid might exist in products but not have any key ingredients
        try:
            prod_len = self.product_index.get_doc_metadata(docid)['length']
        except KeyError:
            prod_len = 0
        
        try:
            ing_len = self.ingredient_index.get_doc_metadata(docid)['length']
        except KeyError:
            ing_len = 0
            
        # Get Average Lengths from pre-calculated stats
        avgdl_prod = self.stats_prod.get('mean_document_length', 0)
        avgdl_ing = self.stats_ing.get('mean_document_length', 0)
        
        score = 0.0
        
        # 2. Iterate through each query term
        for q_term, qtf in query_word_counts.items():
            # CRITICAL FIX: Reset TFs for every term
            tf_prod = 0
            tf_ing = 0
            
            # 3. Retrieve TF specific to each field
            # We cannot use 'doc_word_counts' because it doesn't distinguish between fields.
            # We must scan the postings list for this docid.
            
            # Product Index Lookup
            if q_term in self.product_index.vocabulary:
                postings = self.product_index.get_postings(q_term)
                for pid, freq in postings:
                    if pid == docid:
                        tf_prod += freq
                        # We sum freq here in case of duplicate entries, though usually there's only one.
            
            # Ingredient Index Lookup
            if q_term in self.ingredient_index.vocabulary:
                postings = self.ingredient_index.get_postings(q_term)
                for pid, freq in postings:
                    if pid == docid:
                        tf_ing += freq

            if tf_prod == 0 and tf_ing == 0:
                continue
                
            # 4. Normalize Frequencies (The "F" part of BM25F)
            # Calculate denominator for Product Description
            denom_prod = 1.0
            if avgdl_prod > 0:
                denom_prod = 1 - self.b + self.b * (prod_len / avgdl_prod)
                
            # Calculate denominator for Ingredient Function
            denom_ing = 1.0
            if avgdl_ing > 0:
                denom_ing = 1 - self.b + self.b * (ing_len / avgdl_ing)
            
            # Combine the weighted, normalized frequencies
            # Note: We sum them BEFORE applying the k1 saturation curve
            w_tf = (self.w_prod * tf_prod / denom_prod) + (self.w_ing * tf_ing / denom_ing)
            
            # 5. Apply Saturation (The BM25 part)
            saturation = w_tf / (self.k1 + w_tf)
            
            # 6. Calculate IDF
            # We use the Product Index as the reference for global document frequency
            df = 0
            if q_term in self.product_index.vocabulary:
                df = self.product_index.get_term_metadata(q_term)['doc_frequency']
            
            N = self.stats_prod['number_of_documents']
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            # 7. Query Term Weighting (k3)
            q_norm = ((self.k3 + 1) * qtf) / (self.k3 + qtf)
            
            score += idf * saturation * q_norm
            
        return score
    
    
    
def query(self, query: str) -> list[tuple[int, float]]:
        # 1. Tokenize and filter stopwords
        tokens = self.tokenize(query)
        # Filter out None/stopwords completely for the Counter
        valid_tokens = [t for t in tokens if t not in self.stopwords and t is not None]
        
        query_word_counts = Counter(valid_tokens)
        
        # 2. Build the candidate dictionary
        # Structure: {docid: {term: {'prod': freq, 'ing': freq}}}
        doc_data = defaultdict(lambda: defaultdict(lambda: {'prod': 0, 'ing': 0}))

        for token in set(valid_tokens): # Use set to avoid processing same term twice
            # A. Product Index Postings
            # Use get_postings() instead of direct dict access for safety
            if token in self.product_index.vocabulary:
                prod_postings = self.product_index.get_postings(token)
                for docid, freq in prod_postings:
                    doc_data[docid][token]['prod'] = freq
            
            # B. Ingredient Index Postings
            if token in self.ingredient_index.vocabulary:
                ing_postings = self.ingredient_index.get_postings(token)
                for docid, freq in ing_postings:
                    doc_data[docid][token]['ing'] = freq

        # 3. Score Documents
        all_scores = []
        for doc_id, term_counts in doc_data.items():
            # term_counts is now: {'acne': {'prod': 1, 'ing': 0}, 'cream': {'prod': 5, 'ing': 0}}
            score = self.scorer.score(doc_id, term_counts, query_word_counts)
            if score > 0:
                all_scores.append((doc_id, score))

        # 4. Sort and Return
        sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        return sorted_scores