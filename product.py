products = {
    "docid" : 0,
    "text" : "entire title + description"
}

ingredientfunctions = {
    "fid" : 0,
    "text" : "entire title + description of each ingredient function"
}

"docid =1, cerave moisturizer, func = [0,2,4]"

# functions index
{
    "brightening" : [3,4,5,1002]
}

# products index
{
    "serum" : [0,3,5,6,1002]

}

# concatenate everything and then run BM25 - ONE SINGULAR INDEX

'''
1. Build out both indexes
2. Modifying BM25 to be BM25F
3. docidid -> {title : url: } -> creating a map
4. run relevance tests
5. annotation
'''
