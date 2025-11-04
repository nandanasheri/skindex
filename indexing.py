'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''
import os
import gzip
import json
from enum import Enum
from collections import Counter, defaultdict
from tqdm import tqdm
from document_preprocessor import Tokenizer


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex and PositionalIndex
    BasicInvertedIndex = 'BasicInvertedIndex'

class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        # Define necessary variables
        # We will use them later when we implement the below methods in BasicInvertedIndex and PositionalIndex classes
        self.statistics = {}
        self.statistics['vocab'] = Counter()
        self.statistics["unique_token_count"] = 0
        self.statistics["total_token_count"] = 0
        self.statistics["number_of_documents"] = 0
        self.statistics["mean_document_length"] = 0
        self.vocabulary = set()
        self.document_metadata = {}
        self.index = defaultdict(list)

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
        # TODO: Implement this to add documents to the index
        raise NotImplementedError

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
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
        # TODO: Implement this to fetch a term's postings from the index
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
        # TODO: Implement to fetch a particular document stored in metadata
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
        # TODO: Implement to fetch a particular term stored in metadata
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
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
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
        self._stats_need_update = False

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
        # TODO: Implement this to add documents to the index

        self._stats_need_update = True

        # create basic index
        tempDict = Counter(tokens)
        unique_tokens = 0
        for token,count in tempDict.items():
            if token is not None:
                self.statistics["vocab"][token] += count
                self.index[token].append((docid, count))
                unique_tokens += 1

        # metadata
        self.document_metadata[docid] = {"unique_tokens": unique_tokens, "length": len(tokens)}

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics

        # remove metadata
        if docid in self.document_metadata:
            self.document_metadata.pop(docid, None)
        else:
            raise KeyError

        # delete document from index
        for key in self.index:
            for docEntry in self.index[key]:
                if docEntry[0] == docid:
                    self.index[key].remove(docEntry)
                #TODO NOTE: remove key if len() is 0???

        #TODO NOTE: update statistics as well

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
        # TODO: Implement this to fetch a term's postings from the index
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
        # TODO: Implement to fetch a particular document stored in metadata
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
        # TODO: Implement to fetch a particular term stored in metadata
        return {"term_count": self.statistics["vocab"][term], "doc_frequency": len(self.index[term])}

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
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic

        # Hint: Statistics only need recomputation if the index has been modified since the last calculation. 
        #       This ensure substantial performance improvements.
        if self._stats_need_update == False:
            return self.statistics

        # statistics
        # self.statistics['vocab'].update(tokens)
        self.statistics["total_token_count"] = sum([self.document_metadata[docid]["length"] for docid in self.document_metadata])
        self.statistics["stored_total_token_count"] = self.statistics['vocab'].total()
        self.statistics["number_of_documents"] = len(self.document_metadata)
        self.statistics["mean_document_length"] = self.statistics["total_token_count"]/self.statistics["number_of_documents"]
        self.statistics["unique_token_count"] = len(self.statistics['vocab'])
        self._stats_need_update = False
        return self.statistics

    def save(self, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        filepath = os.path.join(index_directory_name, "index.json")
        os.makedirs(index_directory_name, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({"index": self.index, "metadata": self.document_metadata}, f, indent=4)
    
    def load(self, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        filepath = os.path.join(index_directory_name, "index.json")
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.index = data.get("index", {})
        self.document_metadata = data.get("metadata", {})

        # Convert metadata keys back to integers
        metadataStrKeys = data.get("metadata", {})
        self.document_metadata = {int(key): value for key, value in metadataStrKeys.items()}

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''
    @staticmethod
    def create_index(index_type: IndexType, inci_products_filepath: str,
                     inci_ingredients_functions_filepath: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            inci_products_filepath: The file path to the product dataset
            inci_ingredients_functions_filepath: The file path to the ingredient function dataset
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
        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        if index_type == IndexType.BasicInvertedIndex:
                index = BasicInvertedIndex()

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # NOTE 1: Make sure to support both .jsonl.gz and .jsonl as input

        # NOTE 2: Word frequencies should be calculated prior to removing stop words
        #         Word frequency refers to how many times that word appears in the collection

        if inci_products_filepath.endswith('.gz'):
            openFile = gzip.open
            interaction = 'rt'
        else:
            openFile = open
            interaction = 'r'

        count = 0
        wordFrequencies = Counter()
        doc_tokens = []
        
        # Process Ingredients and Ingredient Function
        with open(inci_ingredients_functions_filepath) as f:
            func_defs_list = json.load(f)

        func_defs_dict = {entry['func_id']: {'title': entry['title'], 'desc': entry['desc']} for entry in func_defs_list}

        # First pass: collect tokens and word frequencies
        with openFile(inci_products_filepath, interaction) as f:
            for line in f:
                if max_docs > -1 and count >= max_docs:
                    break
                product = json.loads(line)
                
                # Gather ingredient function info
                ingredients_text = ""
                for function_id in product.get("key_ingredient_func", []):
                    ingredients_text += func_defs_dict[function_id]["title"] + " " + func_defs_dict[function_id]["desc"] + " "
                
                # Ingredient Title, Ingredient Description, Product Title, Product Description, Product Brand
                concat_text = f"{ingredients_text} {product.get('title', '')} {product.get('desc', '')} {product.get('brand', '')}"

                tokens = document_preprocessor.tokenize(concat_text)
                doc_tokens.append((product["docid"], tokens))
                wordFrequencies.update(tokens)
                count += 1

        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency
        if minimum_word_frequency > 0:
            allowed_tokens = {word for word, freq in wordFrequencies.items() if freq >= minimum_word_frequency}
        else:
            allowed_tokens = set(wordFrequencies.keys())
            
        if stopwords:
            allowed_tokens -= stopwords

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency
        for doc_id, doc_token_list in tqdm(doc_tokens, total=len(doc_tokens)):
            filtered_tokens = [
                token for token in doc_token_list if token in allowed_tokens
            ]
            index.add_doc(doc_id, filtered_tokens)
        index.get_statistics()
        return index