'''
Create Inverted Index for List of Products and List of Ingredient Functions
'''
import os
import gzip
import json
from enum import Enum
from collections import Counter, defaultdict
from tqdm import tqdm
from document_preprocessor import Tokenizer, RegexTokenizer


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex and PositionalIndex
    BasicInvertedIndex = 'BasicInvertedIndex'
    PositionalIndex = 'PositionalIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.

        Note: The following variables are defined to help you store some summary info about your document collection
                for a quick look-up.
              You may also define more variables and/or keys as you see fit.
        Variables:
            statistics: A dictionary, which is the central statistics of the index.
                        Some keys include:
                statistics['vocab']: A counter which keeps track of the token count
                statistics['unique_token_count']: how many unique terms are in the index
                statistics['total_token_count']: how many total tokens are indexed including filterd tokens),
                    i.e., the sum of the lengths of all documents
                statistics['stored_total_token_count']: how many total tokens are indexed excluding filtered tokens
                statistics['number_of_documents']: the number of documents indexed
                statistics['mean_document_length']: the mean number of tokens in a document (including filter tokens)
                ...
                (Add more keys to the statistics dictionary as you see fit)
                
            vocabulary: A set of distinct words that have appeared in the collection
            document_metadata: A dictionary, which keeps track of some important metadata for each document.
                               Assume that we have a document called 'doc1', some keys include:
                document_metadata['doc1']['unique_tokens']: How many unique tokens are in the document (among those not-filtered)
                document_metadata['doc1']['length']: How long the document is in terms of tokens (including those filtered) 
                ...
                (Add more keys to the document_metadata dictionary as you see fit)
            index: A dictionary of class defaultdict, its implemention depends on whether we are using 
                            BasicInvertedIndex or PositionalIndex.
                    BasicInvertedIndex: Store the mapping of terms to their postings
                    PositionalIndex: Each term keeps track of documents and positions of the terms occurring in the document
            
        Example:
            document1 = ['This', 'is' ,'a', 'dog', None]
            document2 = [None, 'This', 'is', 'a', 'cat']

            statistics = {
                'vocab'                     : Counter({'This': 2, 'is': 2, 'a' : 2, 'dog' : 1, 'cat': 1}),
                'unique_token_count'        : 5,
                'total_token_count'         : 10,
                'stored_total_token_count'  : 8,
                'number_of_documents'       : 2,
                'mean_document_length'      : 5
            }

            vocabulary = {'This', 'is', 'a', 'cat', 'dog'}

            document_metadata = {
                'document1': {'unique_tokens': 4, 'length': 5},
                'document2': {'unique_tokens': 4, 'length': 5}
            }

            If BasicInvertedIndex, we store 'term': ((docid, count))
            index = {
                'This': (('document1', 1), ('document2', 1)), 
                'is': (('document1', 1), ('document2', 1)),
                'a': (('document1', 1), ('document2', 1)),
                'dog': (('document1', 1))
                'cat': (('document2', 1))
            }
            If PositionalIndex, we store 'term': ((docid, count, position))
            index = {
                'This': (('document1', 1, 0), ('document2', 1, 1)), 
                'is': (('document1', 1, 1), ('document2', 1, 2)),
                'a': (('document1', 1, 2), ('document2', 1, 3)),
                'dog': (('document1', 1, 3))
                'cat': (('document2', 1, 4))
            }
     
        """
        # Define necessary variables
        # We will use them later when we implement the below methods in BasicInvertedIndex and PositionalIndex classes
        self.statistics = {}   
        self.statistics['vocab'] = Counter()  
        self.statistics['unique_token_count'] = 0
        self.statistics['total_token_count'] = 0
        self.statistics['stored_total_token_count'] = 0
        self.statistics['number_of_documents'] = 0 
        self.statistics['mean_document_length'] = 0 
        self.vocabulary = set()  
        self.document_metadata = {}
        self.index = defaultdict(list)
        self.filtered_word_count = 0
        # store a set of docids
        self.docs = set()
        self.stats_need_update = False       


    # NOTE: The following functions have to be implemented in the two inherited classes and NOT in this class

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include AT LEAST the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic

        # Hint: Statistics only need recomputation if the index has been modified since the last calculation. 
        #       This ensure substantial performance improvements.
        raise NotImplementedError

    def save(self, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # generate frequency of tokens
        doc_counts = Counter(tokens)

        # add to index and vocabulary
        for token in doc_counts:
            self.index[token].append((docid, doc_counts[token]))
            if token not in self.vocabulary:
                self.vocabulary.add(token)

        self.docs.add(docid)
        # update vocab counts with new words
        self.statistics['vocab'].update(doc_counts)
        # update document metadata
        self.document_metadata[docid] = {}
        self.document_metadata[docid]['length'] = len(tokens)

        self.document_metadata[docid]['unique_tokens'] = len(doc_counts)
        # to update statistics later on
        self.stats_need_update = True

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # Implement this to remove a document from the entire index and statistics
        new_index = defaultdict(list)
        removed_tokens = []
        for token in self.index:
            for doc,count in self.index[token]:
                if doc != docid:
                    new_index[token].append((doc, count))
                else:
                    removed_tokens.append(token)
        
        self.index = new_index
        self.docs.remove(docid)
        # update document metadata
        self.document_metadata.pop(docid)
        self.stats_need_update = True

        # update vocab
        removed_token_count = Counter(removed_tokens)
        # update vocab counts by removing frequences of removed tokens
        self.statistics['vocab'] = self.statistics['vocab'] - removed_token_count
        # takes intersection of existing vocab and updated terms to remove any tokens that were unique to docid from the vocabulary
        self.vocabulary = self.vocabulary & set(self.statistics['vocab'])


    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index[term]

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata[doc_id]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # Implement to fetch a particular term stored in metadata
        term_metadata = {}
        term_metadata["term_count"] = self.statistics['vocab'][term]
        term_metadata['doc_frequency'] = len(self.index[term])
        return term_metadata

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include AT LEAST the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic

        # Hint: Statistics only need recomputation if the index has been modified since the last calculation. 
        #       This ensure substantial performance improvements.
        if self.stats_need_update:
            # update statistics
            if None in self.vocabulary:
                self.statistics['unique_token_count'] = len(self.vocabulary) - 1
            else:
                self.statistics['unique_token_count'] = len(self.vocabulary)
            self.statistics['number_of_documents'] = len(self.docs)
            self.statistics['total_token_count'] = sum(self.statistics['vocab'].values())
            # stored total tokens - remove counts of None
            self.statistics['stored_total_token_count'] = self.statistics['total_token_count'] - self.statistics['vocab'][None]


            if self.statistics['number_of_documents'] == 0:
                self.statistics['mean_document_length'] = 0
            else:
                self.statistics['mean_document_length'] = self.statistics['total_token_count'] / (self.statistics['number_of_documents'] * 1.0)

            self.stats_need_update = False

        return self.statistics

    def save(self, index_type:str, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        data = {}

        # convert datatypes like Collections and defaultdict to basic Python dictionaries for json writing
        data['index'] = dict(self.index)
        data['statistics'] = self.statistics
        data['statistics']['vocab'] = dict(data['statistics']['vocab'])
        data['document_metadata'] = self.document_metadata
        data['vocabulary'] = list(self.vocabulary)
        data['docs'] = list(self.docs)
        data['status_needs_update'] = self.stats_need_update
        
        # create directory
        if not os.path.exists(index_directory_name):
            os.mkdir(index_directory_name)
        # concatenate full filepath
        filepath = os.path.join(index_directory_name, f"inverted_index_{index_type}.json")

        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, index_type:str, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        filepath = os.path.join(index_directory_name, f"inverted_index_{index_type}.json")

        with open(filepath, "r") as f:
            data = json.load(f)
            self.index = defaultdict(list, data['index'])
            self.statistics = data['statistics']
            self.statistics['vocab'] = Counter(self.statistics['vocab'])
            doc_data = data['document_metadata']
            self.document_metadata = {}
            for key in doc_data:
                self.document_metadata[int(key)] = doc_data[key]
            self.vocabulary = set(data['vocabulary'])

            self.docs = set(data['docs'])
            self.stats_need_update = data['status_needs_update'] 

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''
    @staticmethod
    def create_index(index_type: IndexType, product_dataset_path: str,
                     function_dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int,
                     max_docs: int = -1) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
                Setting a value that is less than 1 (e.g. 0 or negative) will result in all documents being indexed.

        Returns:
            An inverted index
        '''
        
        # Create index as specified
        product_index = None
        function_index = None
        if index_type == IndexType.BasicInvertedIndex:
            product_index = BasicInvertedIndex()
            function_index = BasicInvertedIndex()
        
        # set flags for stop words, minimum word frequency and max number of documents processed
        is_stopwords = True
        if not stopwords or len(stopwords) == 0:
            is_stopwords = False
    
        # Create a Mapping from func_id to tokenized and processed text where text = title + desc
        i = 0
        func_to_tokens = {}
        if function_dataset_path.endswith("jsonl"):
            with open(function_dataset_path, "r") as f:
                # go through each json line
                for line in f:
                    if i == max_docs:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    eachdoc = json.loads(line)
                    # concatenate two columns together
                    text = eachdoc['title'] + " " + eachdoc['desc']
                    # if there is no minimum frequency, postprocess every token
                    func_tokens = document_preprocessor.tokenize(text)
                    if not is_stopwords:
                        func_to_tokens[eachdoc['func_id']] = func_tokens
                    else:
                        # remove stopwords and then postprocess 
                        tokens_without_stopwords = []
                        for token in func_tokens:
                            if token not in stopwords:
                                tokens_without_stopwords.append(token)
                            else:
                                tokens_without_stopwords.append(None)
                        func_to_tokens[eachdoc['func_id']] = tokens_without_stopwords
                    i += 1

        # load in data for jsonl
        i = 0
        if product_dataset_path.endswith("jsonl"):
            with open(product_dataset_path, "r") as f:
                # go through each json line
                for line in f:
                    if i == max_docs:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    eachdoc = json.loads(line)
                    text = eachdoc['title'] + " " + eachdoc['brand'] + eachdoc['desc']
                    # if there is no minimum frequency, postprocess every token
                    product_tokens = document_preprocessor.tokenize(text)
                    
                    # no stopwords - postprocess and add to product_index
                    if (not is_stopwords):
                        product_index.add_doc(eachdoc['docid'], product_tokens)
                    
                    # if stopwords should be removed
                    else:
                        tokens_without_stopwords = []
                        for token in product_tokens:
                            if token not in stopwords:
                                tokens_without_stopwords.append(token)
                            else:
                                tokens_without_stopwords.append(None)
                        # only post process tokens that aren't stopwords
                        product_index.add_doc(eachdoc['docid'], tokens_without_stopwords) 

                    if 'key_ingredient_func' in eachdoc:
                        for func in eachdoc['key_ingredient_func']:
                            function_index.add_doc(eachdoc['docid'], func_to_tokens[func])
                        i += 1

        # Return both indexes
        return product_index, function_index

if __name__ == '__main__':
    pass