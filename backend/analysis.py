import csv
import numpy as np
import re
import math
from collections import Counter
from typing import List, Tuple, Dict
"""
File for putting analysis related functions
"""

#####minimum edit#################################################################
def insertion_cost(message, j):
    return 1


def deletion_cost(query, i):
    return 1


def substitution_cost(query, message, i, j):
    if query[i - 1] == message[j - 1]:
        return 0
    else:
        return 1
def get_titleidx(titles, inverse):
   idx = []
   for t in titles:
      idx.append(inverse[t])
   return idx
      

def edit_matrix(query, message, ins_cost_func, del_cost_func, sub_cost_func):
    """Calculates the edit matrix

    Arguments
    =========

    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit matrix {(i,j): int}
    """

    m = len(query) + 1
    n = len(message) + 1

    chart = {(0, 0): 0}
    for i in range(1, m):
        chart[i, 0] = chart[i - 1, 0] + del_cost_func(query, i)
    for j in range(1, n):
        chart[0, j] = chart[0, j - 1] + ins_cost_func(message, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i - 1, j] + del_cost_func(query, i),
                chart[i, j - 1] + ins_cost_func(message, j),
                chart[i - 1, j - 1] + sub_cost_func(query, message, i, j),
            )
    return chart
def edit_distance(
    query: str, message: str, ins_cost_func: int, del_cost_func: int, sub_cost_func: int
) -> int:
    """Finds the edit distance between a query and a message using the edit matrix

    Arguments
    =========
    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit cost (int)
    """

    query = query.lower()
    m = len(query)
    message = message.lower()
    n = len(message)


    # TODO-1.1
    mat = edit_matrix(query, message, ins_cost_func, del_cost_func, 
    sub_cost_func)
    
    
    return mat[(m,n)]
    
    

def edit_distance_search(
    query: str,
    msgs: List[dict],
    ins_cost_func: int,
    del_cost_func: int,
    sub_cost_func: int,
) -> List[Tuple[int, str]]:
    """Edit distance search

    Arguments
    =========
    query: string,
        The query we are looking for.

    msgs: list of dicts,
        Each message in this list has a 'text' field with
        the raw document.

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns
    =======
    result: list of (score, message) tuples.
        The result list is sorted by score such that the closest match
        is the top result in the list.

    """
    # TODO-1.2
    scores = []
    mess = []
    final = []

    for m in msgs:
      scores.append(edit_distance(query, m,  ins_cost_func, del_cost_func, sub_cost_func))
      mess.append(m)

    scores = np.array(scores)
    mess = np.array(mess)

    s = np.argsort(scores)
    mess = mess[s]
    scores = scores[s]
    

    

    for i in range(10):
      final.append(mess[i])
    return final

######minimum edit########################################################################
def tokenize(text: str):
    low_text = text.lower()
    pattern = r'[a-z]+'
    word_list = re.findall(pattern, low_text)
    return word_list


def build_doc_inverted_index(doc_lst):
  """Builds an inverted index from the titles.

  Arguments
  =========

  title_lst: list of titles.
      Each title in the list corresponds to a title in the data set

  Returns
  =======

  inverted_index: dict
  """
  inv_idx = {}
  for doc_idx in range(len(doc_lst)):
     inv_idx[doc_lst[doc_idx]] = doc_idx
  return inv_idx

def build_token_inverted_index(doc_lst: list, doc_inv_idx: dict) -> dict:
    """Builds an inverted index from the documents.
    

    Arguments
    =========

    title_lst: list of titles.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    title_inv_idx: dict.
        Mapping from titles to integers that represent the index of that title.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> Make a nice example here

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]
    """
    token_inv_idx = {}
    for doc in doc_lst:
        doc_idx = doc_inv_idx[doc]
        doc_tokenized = tokenize(doc)
        doc_tokenized_set = list(set(doc_tokenized))
        for tok in doc_tokenized_set:
           if tok in token_inv_idx:
              token_inv_idx[tok].append(doc_idx)
           else:
              token_inv_idx[tok] = [doc_idx]
              
    return token_inv_idx


def build_token_inverted_index_with_freq(doc_lst: list, doc_inv_idx: dict) -> dict:
    """Builds an inverted index from the documents.
    

    Arguments
    =========

    title_lst: list of titles.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    title_inv_idx: dict.
        Mapping from titles to integers that represent the index of that title.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> Make a nice example here

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]
    """
    token_inv_idx = {}
    for doc in doc_lst:
        doc_idx = doc_inv_idx[doc]
        doc_tokenized = tokenize(doc)
        doc_tokenized_set = list(set(doc_tokenized))
        for tok in doc_tokenized_set:
           if tok in token_inv_idx:
              token_inv_idx[tok].append((doc_idx, doc_tokenized.count(tok)))
           else:
              token_inv_idx[tok] = [(doc_idx, doc_tokenized.count(tok))]
              
    return token_inv_idx


def boolean_search(query, token_inv_idx : dict, num_docs : int):
  """Search the collection of documents that contain each token
    of the given query for the given query.

  Arguments
  =========

  query: string,
      The word we are searching for in our documents.

  token_inv_idx: dict,
      For each term, the index contains
      a sorted list of tuples doc_id
      such that tuples with smaller doc_ids appear first:
      inverted_index[term] = [d1, d2, ...]

  num_docs: the number of documents.


  Returns
  =======

  results: list of ints
      Sorted List of results (in increasing order) such that every element is a `doc_id`
      that points to a document that satisfies the boolean
      expression of the query.

  """
  query_tok = tokenize(query)
  results = set(range(num_docs))
  for tok in query_tok:
    if tok not in token_inv_idx:
      return []
    results = results.intersection(set(token_inv_idx[tok]))
  return list(results)

def word_counts(str_query : str) -> dict:
   """
   Returns a dictionary containing all words that appear in the query/string;
   Each word is mapped to a count of how many times it appears in the query/string.
   In other words, result[w] = the term frequency of w in the query/string.
   """
   result = {}
   tokenized = tokenize(str_query)
   tokenize_set = set(tokenized)
   for tok in tokenize_set:
      result[tok] = tokenized.count(tok)
   return result

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    idf_dict = {}
    
    for term in inv_idx:
      df = len(inv_idx[term])
      if df >= min_df and df / n_docs < max_df_ratio:
        idf_in = n_docs / (1 + df)
        idf_dict[term] = math.log(idf_in, 2)

    return idf_dict

def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    doc_dict = {}

    for word, idf_val in idf.items():
        for doc_id, tf in index.get(word, []):
            doc_words = doc_dict.get(doc_id, [])
            doc_words.append((tf, idf_val))
            doc_dict[doc_id] = doc_words

    norms_array = np.zeros(n_docs)

    for doc_id in range(n_docs):
        norm_sum = 0
        for tf, idf_val in doc_dict.get(doc_id, []):
            norm_sum += (tf * idf_val) ** 2
        norms_array[doc_id] = math.sqrt(norm_sum)

    return norms_array

def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    doc_scores = {}

    for word in index:
      word_docs = index[word]
      if word in query_word_counts:
        for doc, word_frequency in word_docs:
          dot = word_frequency * query_word_counts[word]
          dot *= idf[word]**2
          doc_scores[doc] = doc_scores.get(doc, 0) + dot
    
    return doc_scores

def index_search(
    query: str,
    index: dict,
    idf,
    doc_norms,
    score_func,
    tokenizer
) -> list:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    Note:

    """
    
    results = []
    query = query.lower()
    tokens = tokenizer(query)
    counts = Counter(tokens)
    query_w_counts = dict(counts)

    query_norm_m = 0
    for token in tokens:
      if token in idf:
        prd_sqr = (idf[token] * query_w_counts[token]) ** 2
        query_norm_m+=prd_sqr

    num_vals= score_func(query_w_counts, index, idf)

    for doc in num_vals:
        cos_sim = num_vals[doc] / (math.sqrt(query_norm_m) * doc_norms[doc])
        results.append((cos_sim, doc))

    return sorted(results,key=lambda x:x[0],reverse=True)


def get_doc_rankings(query, doc_lst):
  """
  Returs the ranked documents
  """
  CUTOFF = 10
  query = query.lower()
  doc_inv_idx = build_doc_inverted_index(doc_lst)
  tok_inv_idx = build_token_inverted_index_with_freq(doc_lst, doc_inv_idx)
  idf_list = compute_idf(tok_inv_idx, len(doc_lst), 0, 1)  # maybe remove limits?
  doc_norms = compute_doc_norms(tok_inv_idx, idf_list, len(doc_lst))
  score_func = accumulate_dot_scores
  results = index_search(query, tok_inv_idx, idf_list, doc_norms, score_func, tokenize)
  idx_results = [doc_id for _,doc_id in results[:CUTOFF]]
  return idx_results