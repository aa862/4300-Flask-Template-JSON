import csv
import numpy as np
import re
# from nltk.tokenize import TreeBankWordTokenizer
"""
File for putting analysis related functions
"""

# treebank_tokenizer = TreeBankWordTokenizer()

def tokenize(text: str):
    # print(text)
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
  # unique_docs_lst = list(set(doc_lst))

  # for doc_idx in range(len(unique_docs_lst)):
  for doc_idx in range(len(doc_lst)):
     inv_idx[doc_lst[doc_idx]] = doc_idx
    # inv_idx[unique_docs_lst[doc_idx]] = doc_idx
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
  # print(type(query))
  query_tok = tokenize(query)
  results = set(range(num_docs))
  # results = set(token_inv_idx[query_tok[0]])
  # print("RESULTS: " + str(results))
  # print(len(query_tok))
  # print(query_tok)
  for tok in query_tok:
    #  print(tok)
    #  print(set(token_inv_idx[tok]))
    # print(token_inv_idx[tok])
    if tok not in token_inv_idx:
      return []
    results = results.intersection(set(token_inv_idx[tok]))
    # print(len(results))
  return list(results)


def get_sim(mov1, mov2):
  # TODO
  pass


def cossim():
  # TODO
  pass