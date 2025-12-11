"""
Document Preprocessor for Tokenization
"""
from nltk.tokenize import RegexpTokenizer, MWETokenizer
# Import additional modules here (if necessary)
# import spacy

# Trained Pipeline for English optimized on CPU
# nlp = spacy.load("en_core_web_sm")
# # for named entity recognition
# nlp.add_pipe("merge_entities")

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # lowercase - boolean that determines tokenization
        # multiword_exp - set of the multiword expressions
        self.lowercase = lowercase
        
        mwe_tokens = []
        if multiword_expressions:
            self.multiword_exp = set(multiword_expressions)
            for exp in multiword_expressions:
                mwe_tokens.append((exp.split()))
        else:
            self.multiword_exp = set()

        self.mwe_tokenizer = MWETokenizer(mwe_tokens, separator=" ")

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition

        Examples:
            If lowercase, "Taylor" "Swift" -> "taylor" "swift"
            If "Taylor Swift" in multiword_expressions, "Taylor" "Swift" -> "Taylor Swift"
        """
        
        postprocess_tokens = []
        # no need to convert to lowercase and no multiword expressions, just return as is
        if not self.lowercase and not self.multiword_exp:
            return input_tokens
        
        # Support for lower-casing and multi-word expressions
        for token in input_tokens:
            if not token:
                continue
            if self.lowercase:
                postprocess_tokens.append(token.lower())
            else:
                postprocess_tokens.append(token)

        if self.multiword_exp:
            postprocess_tokens = self.mwe_tokenizer.tokenize(postprocess_tokens)
        return postprocess_tokens

    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # NOTE: You should implement this in a subclass, not here
        raise NotImplementedError(
            'tokenize() is not implemented in the base class; please use a subclass')


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        The Natural Language Toolkit (NLTK) is a Python package for natural language processing.
        To learn more, visit https://pypi.org/project/nltk/

        Installation Instructions:
            Please visit https://spacy.io/usage
            It is recommended to install packages in a virtual environment.
            Here is an example to do so:
                $ python -m venv [your python virtual enviroment]
                $ source [your python virtual enviroment]/bin/activate # or [your python virtual environment]\Scripts\activate on Windows
                $ pip install -U nltk
                
        Your tasks:
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # Token Regex - save as a field of the class
        self.token_regex = token_regex
        # Initialize the NLTK's RegexpTokenizer 
        self.tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # tokenize using the NLTK's tokenizer for Regex and then post process
        input_tokens = self.tokenizer.tokenize(text)
        return self.postprocess(input_tokens)

    
# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    pass