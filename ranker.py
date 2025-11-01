"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
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
    # TODO: This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document (Not needed for HW1)
        """
        self.index = index
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

        # 2.1 For each token in the tokenized query, find out all documents that contain it and counting its frequency within each document.
        # Hint 1: To understand why we need the info above, pay attention to docid and doc_word_counts, 
        #    located in the score() function within the RelevanceScorer class
        # Hint 2: defaultdict(Counter) works well in this case, where we store {docids : {query_tokens : counts}}, 
        #         or you may choose other approaches

        # 2.2 Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes) 
        #        for each relevant document determined in 2.1

        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]

        query = self.tokenize(query)
        # query = [q.lower() for q in query]
        results = []
        tokens = []

        # 1 remove stopwords
        for word in query:
            if word not in self.stopwords:
                tokens.append(word)

        # 2.1 count of tokens in each doc
        doc_count_tokens = defaultdict(Counter)
        for token in tokens:
            documents = self.index.index[token]
            for docId, count in documents:
                doc_count_tokens[docId][token] += count

        # 2.2 generate score per doc
        for docId in doc_count_tokens:
            score = self.scorer.score(docId, doc_count_tokens[docId], Counter(query))
            results.append((docId, score))

        # PROJECT: added this part to get 
        all_docids = set(self.index.document_metadata.keys())
        scored_docids = set(doc_count_tokens.keys())
        missing_docids = all_docids - scored_docids

        for docId in missing_docids:
            results.append((docId, 0.0))

        return sorted(results, key=lambda x: x[1], reverse=True)


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


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO: Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        total = 0
        for token in query_word_counts:
            total += doc_word_counts.get(token, 0) * query_word_counts.get(token, 0)

        # 2. Return the score
        return total


# TODO: Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid)["length"]
        mu = self.parameters['mu']

        # 2. Compute additional terms to use in algorithm
        score = 0
        for q_term in query_word_counts:
            if q_term and q_term in self.index.index:
                #postings = self.index.get_postings(q_term)
                doc_tf = doc_word_counts[q_term]

                if doc_tf > 0:
                    query_tf = query_word_counts[q_term]
                    p_wc = self.index.get_term_metadata(q_term)["term_count"] / self.index.get_statistics()['total_token_count']
                    tfidf = np.log(1 + (doc_tf / (mu * p_wc)))

                    score += (query_tf * tfidf)

        # 3. For all query_parts, compute score
        score = score + len(query_word_counts) * np.log(mu / (doc_len + mu))

        # 4. Return the score
        return score

# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        N = self.index.statistics["number_of_documents"] # N
        doc_len = self.index.get_doc_metadata(docid)["length"] # |d|
        avg_doc_len = self.index.statistics["mean_document_length"] # avdl

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        score = 0
        for word in query_word_counts:
            if word in doc_word_counts:
                wordCountInDoc = (doc_word_counts[word]) #c(w)
                docCountWithWord = len(self.index.index[word]) # df(w)

                wordCountInQuery = query_word_counts[word] # c(w,q)

                idf = np.log((N - docCountWithWord + 0.5) / (docCountWithWord + 0.5))
                tf = ((self.k1 + 1) * wordCountInDoc) / ((self.k1 * (1 - self.b + (self.b*(doc_len/avg_doc_len)))) + wordCountInDoc)
                qtf = ((self.k3 + 1) * wordCountInQuery) / (self.k3 + wordCountInQuery)

                score += (idf * tf * qtf)

        # 4. Return score
        return score.item()


# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        D = self.index.statistics["number_of_documents"] # |D|
        doc_len = self.index.get_doc_metadata(docid)["length"] # |d|
        avg_doc_len = self.index.statistics["mean_document_length"] # avdl
        score = 0

        # 2. Compute additional terms to use in algorithm
        for word in doc_word_counts:
            c_wq = query_word_counts[word] # c(w,q)
            c_wd = doc_word_counts[word] # c(w,d)
            dfw = len(self.index.index[word]) # df(w)

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
            qtf = c_wq
            tf = (1 + np.log(1 + np.log(c_wd))) / (1 - self.b + (self.b * (doc_len/avg_doc_len)))
            idf = np.log((D + 1)/dfw)

            score += (qtf * tf * idf)

        # 4. Return the score
        return score.item()


# TODO: Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        D = self.index.statistics["number_of_documents"] # |D|
        score = 0

        # 2. Compute additional terms to use in algorithm
        for word in query_word_counts:
            if word in doc_word_counts:
                c_wd = doc_word_counts[word] # c(w)
                df_w = len(self.index.index[word]) # df(w)

        # 3. For all query parts, compute the TF and IDF to get a score
                tf = np.log(c_wd + 1)
                idf = np.log(D/df_w) + 1
                score += tf * idf

        # 4. Return the score
        return score.item()
