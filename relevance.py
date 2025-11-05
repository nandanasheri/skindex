"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.test.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (non-relevant) to 5 (very relevant). When you calculate MAP, treat 4s and 5s as relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant.
"""
import math
import csv
from tqdm import tqdm
import numpy as np


def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    score = 0.0
    count_relevant = 0
    for index, item in enumerate(search_result_relevances[:cut_off]):
        if item == 1:
            count_relevant += 1
            precision = np.divide(count_relevant, index + 1)
            score += precision
    if count_relevant == 0:
        return 0
    return np.divide(score, sum(search_result_relevances))        


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    search_result_relevances = search_result_relevances[:cut_off]
    ideal_relevance_score_ordering = ideal_relevance_score_ordering[:cut_off]
    # DCG
    dcg = 0.0
    for i, rel in enumerate(search_result_relevances, start=1):
        if i == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(i)

    # IDCG
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance_score_ordering, start=1):
        if i == 1:
            idcg += rel
        else:
            idcg += rel / math.log2(i)

    return dcg / idcg if idcg > 0 else 0.0

    # # Calculate DCG Score
    # dcg_score = 0.0
    # for index, item in enumerate(search_result_relevances[:cut_off]):
    #     # dcg_score += np.divide(item, np.log2(index + 2))
    #     if index == 1:
    #         dcg_score = 
            
    # # Calculate IDCG Score
    # idcg_score = 0.0
    # for index, item in enumerate(ideal_relevance_score_ordering[:cut_off]):
    #     idcg_score += np.divide(item, np.log2(index + 2))
    
    # # Calculate NDCG Score
    # if idcg_score == 0:
    #     return 0.0
    # score = np.divide(dcg_score, idcg_score)
    # return score


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: Treat search results from your ranking function that are not listed in the relevance_data_filename as non-relevant

    # NOTE: NDCG can use any scoring range, so no conversion is needed.

    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    with open(relevance_data_filename, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        ground_truth = dict()
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            query_text = row['query']
            docid = int(row['docid'])
            rel = int(row['rel'])
            if query_text not in ground_truth:
                ground_truth[query_text] = {}
            ground_truth[query_text][docid] = rel
        
        map_list = []
        ndcg_list = []
        for query_text in ground_truth:
            ranked_documents = ranker.query(query_text)
            ranked_docid = [docid for docid,_ in ranked_documents[:10]]
            
            # Convert relevant scores to binary judgements for MAP
            rel_binary = [ 1 if ground_truth[query_text].get(docid,0) >= 4 else 0 for docid in ranked_docid]
            # For NDCG (multi-level relevance)
            rel_real = [ ground_truth[query_text].get(docid, 0) for docid in ranked_docid ]
            # For Ideal NDCG 
            rel_ideal = sorted(ground_truth[query_text].values(), reverse=True)[:10]
            
            map = map_score(rel_binary)
            ndcg = ndcg_score(rel_real, rel_ideal)
            
            map_list.append(map)
            ndcg_list.append(ndcg)
        
        average_map = np.mean(map_list)
        average_ndcg = np.mean(ndcg_list)
        
        return {'map': average_map, 'ndcg': average_ndcg, 'map_list': map_list, 'ndcg_list': ndcg_list}


if __name__ == '__main__':
    pass