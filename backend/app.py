import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import analysis

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'compressed_df.json')

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
    # data = json.load(file)
    # user_df = pd.DataFrame(data['user'])
    # age_occ_df = pd.DataFrame(data['age_occupation'])
    # compressed_df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)

### Constants ###

# the number of results to print on the screen.
NUM_RESULTS = 50
# the fields of the json to print
FIELDS_TO_PRINT = ['title', 'authors', 'ban_info', 'genres', 'ratings', 'summary']

### Search Helper Functions ###

def build_ban_freq_dict(ban_info_str: str) -> dict:
    """
    Returns a dictionary of mappings of states to the
    number of regions in which the book is banned in that state.

    Precondition: The parameter ``ban_info_str`` is a str
    representing the ban information, where information for
    the regions a book is banned in are delimited by ";",
    and the information for each specific region ban is
    delimited by commas, and the state is the second
    element of that delimited list.
    """
    state_freq_dict = {} 
    
    try:
        reg_info_lst = ban_info_str.strip().split(";")[:-1] # do we miss final ban here sometimes?

    except:
        print(ban_info_str)
    
    else:
        for reg_info in reg_info_lst:
            data_fields = reg_info.split(",")
            state = data_fields[1].strip()
            if state in state_freq_dict:
                state_freq_dict[state] += 1
            else:
                state_freq_dict[state] = 1
    return state_freq_dict

def build_new_ban_info_col(df_col):
    """
    Returns a list of strings to be used as the new column
    for df_col "ban_info" column of the matches dataframe.

    Precondition: ``df_col`` is a column of data from a dataframe.

    States and frequency of banning is extracted from each row
    of ``df_col``, and a new list with each element in the format:
        state, (frequency_of_bans_in_state)
    is returned.
    """
    # dictionaries of mappings of state to frequency of banning
    ban_freq_dict = {}
    # list of dictionaries of mappings of state to frequency of banning (list of ban_freq_dicts)
    ban_freq_dict_lst = []
    # list of strings in format: state, (frequency_of_bans_in_state)
    new_lst = []
    for ban_info_str in df_col:
        ban_freq_dict = build_ban_freq_dict(ban_info_str)
        ban_freq_dict_lst.append(ban_freq_dict)
    for freq_dict in ban_freq_dict_lst:
        state_ban_info_str = ""
        for state in freq_dict:
            body_str = state + " (" + str(freq_dict[state]) + "), "
            state_ban_info_str = state_ban_info_str + body_str
        new_lst.append(state_ban_info_str[:-2])
    return new_lst

def build_new_authors_col(df_col):
    """
    Returns a list of strings to be used as the new column
    for df_col "authors" column of the matches dataframe.

    The new authors column is a reordering of the original author
    names from "lastname, firstname" to "firstname lastname"

    Precondition: ``df_col`` is a column of data from a dataframe.
    """
    a = []

    for author in df_col:
        new_a = author
        str_lst = author.split(",")
        if len(str_lst) == 2:
            last = str_lst[0]
            first = str_lst[1]
            new_a = first + " " + last
        new_a = new_a.strip()
        a.append(new_a)

    return a

### Searches ###

def boolean_sim_search(query, docs):
    """
    Returns a list of the most similar documents to the query
    using boolean AND search.
    
    Similarity is determined using a boolean AND search between
    the words of the query.

    Precondition: ``docs`` is an iterable object
    """
    title_inv_idx = analysis.build_doc_inverted_index(docs)
    tok_inv_idx = analysis.build_token_inverted_index(docs, title_inv_idx)
    results = analysis.boolean_search(query, tok_inv_idx, len(docs), NUM_RESULTS)
    return results

def edit_dist_search(query, docs):
    """
    Returns a list of the most similar documents to the query
    using edit distance.
    
    Similarity is determined using a boolean AND search between
    the words of the query.

    Precondition: ``docs`` is an iterable object
    """
    title_inv_idx = analysis.build_doc_inverted_index(docs)
    # tok_inv_idx = analysis.build_token_inverted_index(doc_lst, title_inv_idx)
    results = analysis.edit_distance_search(query, docs, analysis.insertion_cost, analysis.deletion_cost, analysis.substitution_cost)
    
    results = analysis.get_titleidx(results,title_inv_idx)
    return results

def cossim_sim_search(query, docs):
    """
    Returns a list of the most similar documents to the query
    using cosine similarity.
    
    Similarity is determined using cosine similarity
    between the query and the summaries of all the books.

    Precondition: ``docs`` is an iterable object
    """
    # reviews = df["reviews"].fillna('')
    # print("len(reviews):")
    # print(len(reviews))
    # print("len(lst_blurb):")
    # print(len(lst_blurb))

    # cossim_results = analysis.get_doc_rankings(query, lst_blurb, NUM_RESULTS)
    cossim_results = analysis.get_doc_rankings(query, docs, NUM_RESULTS)
    # print("cossim_results:")
    # print(cossim_results)
    # print("here 2")
    return cossim_results

def svd_sim_search(query, docs):
    """
    Returns a list of the most similar documents to the query using SVD.
    
    Similarity is determined using SVD similarity
    between the query and the summaries of all the books.

    Precondition: ``docs`` is an iterable object
    """
    docs_compressed_normed, words_compressed_normed, query_vec = analysis.svd_analysis(docs, query)
    # MAKE SURE TO TRANSPOSE words_compressed_normed!!!
    top_matches_lst = analysis.closest_docs_to_query(docs_compressed_normed, words_compressed_normed.T, query_vec, NUM_RESULTS)
    return top_matches_lst

### Production Conversion ###

def convert_to_json(matches_lst: list, genre = "", state="" ):
    """
    Returns the json of information for the documents in ``matches_lst``.

    Each document will have information from the fields in ``FIELDS_TO_PRINT``.
    """
    df = pd.read_csv("data/finalized_books.csv")
    matches_filtered = matches_lst
    if genre:
    #    print("IN GENREEE")
       matches_filtered = filter_genre(matches_lst,genre,df)
    #    print(f"MATCHES FILTERED GENRE: {matches_filtered}")
    if state:
        # print("IN STATEE")
        state_dict = filter_state_helper(df)
        
        # print(f"STATE_DICT: {state_dict}")
        # print("STATEEE")
        # print(state)
        matches_filtered = list(set(matches_filtered).intersection(set(state_dict.get(state,set()))))
        # print(f"MATCHES FILTERED STATE: {matches_filtered}")
        
    # print(f"MATCHES FILTERED BEFORE: {matches_filtered}")
    matches_filtered = df.iloc[matches_filtered]
    matches_filtered = matches_filtered[FIELDS_TO_PRINT]
    # print(f"MATCHES FILTERED AFTER: {matches_filtered}")
    # construct new columns for data that needs to be displayed differently
    all_authors_info = matches_filtered['authors']
    matches_filtered["authors"] = build_new_authors_col(all_authors_info)

    all_ban_info = matches_filtered['ban_info']
    matches_filtered['ban_info'] = build_new_ban_info_col(all_ban_info)

    # print("first rating:")
    # print(matches_filtered["ratings"])

    jsonified = matches_filtered.to_json(orient='records')
    return jsonified

def filter_genre(match_list: list, genre:str, df:pd.DataFrame)->list:
    """
    Returns an instance of match_list where only the indices with the specified
    genre are returned.

    match_list: the initial list of indices
    genre: the wanted genre
    df: the dataframe of all the documents, must have column 'Genre' in it.
    """
   
    genre_only_lst = []
    for index,  row in df.iterrows():
        genres_in_book = row['genres'].strip().split(', ')
        # print(f"GENRESINBOOK {genres_in_book}")
        if genre in genres_in_book:
            genre_only_lst.append(index)
    # print(f" GENRE is : {genre}")
    # print(f"GenreLIst {genre_only_lst}")
    # print(f" MatchList {match_list} Length : {len(match_list)}")
    temp = list(set(genre_only_lst).intersection(set(match_list)))
    # print(f"TEMP : {temp}")
    return temp

def filter_state_helper(df):
    """
    returns a dictionary that maps states to the indices of banned books in that
    state

    df: the dataframe with column state
    """
    result = {}

    for index, row in df.iterrows():
        try:
            # print("In TRY")
            reg_info_lst = row['ban_info'].strip().split(";")[:-1]
        except:
            # print("In EXCEPT")
            raise Exception ("Couldnt split file")
        else:
            #print("In ELSE")
            for reg_info in reg_info_lst:
                data_fields = reg_info.split(",")
                state = data_fields[1].strip()
                result[state] = result.get(state, [])  + [index]

    return result 


### Funcions to be called in HTML ###

def title_search(query, sim_measure_code):
    """
    Returns the a JSON with the information on the documents
    with the ``NUM_RESULTS`` most similar titles
    to ``query`` using the similarity measure represented
    by ``sim_measure_code``.
    
    Each document will have information from the fields in ``FIELDS_TO_PRINT``.

    If sim_measure_code =
    - 0: boolean similarity between words of query
    - 1: edit distance similarity between titles of documents
    """
    df = pd.read_csv("data/finalized_books.csv")

    docs = df['title']

    matches_lst = []
    if sim_measure_code == 0:
        matches_lst = boolean_sim_search(query, docs)
        if len(matches_lst) == 0:
            matches_lst = edit_dist_search(query, docs)
    if sim_measure_code == 1:
        matches_lst = edit_dist_search(query, docs)
    return convert_to_json(matches_lst)

def theme_search(query="", sim_measure_code=0, state="", genre=""):
    """
    Returns the a JSON with the information on the documents
    with the ``NUM_RESULTS`` most similar themes
    to ``query`` using the similarity measure represented
    by ``sim_measure_code``.
    
    Each document will have information from the fields in ``FIELDS_TO_PRINT``.

    If sim_measure_code =
    - 0: cosine similarity: 
    - 1: SVD similarity
    """
    df = pd.read_csv("data/finalized_books.csv")

    cossim_docs = df["summary"].fillna('')
    svd_docs = df["reviews"].fillna('')

    # state_dict = filter_state_helper(df)
    # genre_lst = df["genres"]
    # for idx in range(len(lst_reviews)):
    #     lst_reviews[idx] += genre_lst[idx]

    matches_lst = []
    if sim_measure_code == 0:
        # print(f"QUERY: {query}")
        matches_lst = cossim_sim_search(query, cossim_docs)
        # print(f"MATCHES LIST COSSIM: {matches_lst}")
    if sim_measure_code == 1:
        matches_lst = svd_sim_search(query, svd_docs)

    if query == "":
        matches_lst = list(range(len(df)))

    return convert_to_json(matches_lst,genre,state)

@app.route("/")
def home():
    return render_template('index.html',title="sample html")

@app.route("/titles")
def titles_search():
    text = request.args.get("title")
    print("title search")
    return title_search(text, 0)

@app.route("/books")
def books_search():
    text = request.args.get("title")
    genre = request.args.get("genre")
    state = request.args.get("state")
    # do request.args to get the state or genre NOTE: they might be null 
    # print("theme search blurbs")
    print("Blurbbbbbbbb")
    return theme_search(text, 0,state,genre)
@app.route("/reviews")
def reviews_search():
    print("reviews search")
    text = request.args.get("title")
    genre = request.args.get("genre")
    state = request.args.get("state")
    
    # do request.args to get the state or genre NOTE: they might be null 
    # print("theme search reviews")
    return theme_search(text, 1 ,state,genre)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5002)
