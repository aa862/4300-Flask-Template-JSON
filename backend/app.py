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
NUM_RESULTS_TO_PRINT = 10
# the fields of the json to print
FIELDS_TO_PRINT = ['title','authors','ban_info', 'summary']

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
    reg_info_lst = ban_info_str.strip().split(";")[:-1] # do we miss final ban here sometimes?
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
    for df_col "ban_info" column of the matched dataframe.

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


### Searches ###

def title_search(query):
    """
    Returns a json of the most similar documents to the query.
    
    Similarity is determined using a boolean AND search between
    the words of the query.
    """
    df = pd.read_csv("final1.csv")

    pd_title = df['title']
    title_inv_idx = analysis.build_doc_inverted_index(pd_title)
    tok_inv_idx = analysis.build_token_inverted_index(pd_title, title_inv_idx)
    results = analysis.boolean_search(query, tok_inv_idx, len(pd_title))
    
    matches_filtered = df.iloc[results]
    matches_filtered = matches_filtered[FIELDS_TO_PRINT]

    all_ban_info = matches_filtered['ban_info']
    matches_filtered['ban_info'] = build_new_ban_info_col(all_ban_info)

    jsonified = matches_filtered.to_json(orient='records')
    return jsonified

def theme_search_cossim(query):
    """
    Returns a json of the most similar documents to the query.
    
    Similarity is determined using cosine similarity
    between the query and the summaries of all the books.
    """
    df = pd.read_csv("final1.csv")
    
    nan_variable = "summary"
    df_cleaned = df[nan_variable].fillna('')
    lst_blurb = df_cleaned

    cossim_results = analysis.get_doc_rankings(query, lst_blurb, NUM_RESULTS_TO_PRINT)
    matches_filtered = df.iloc[cossim_results]
    matches_filtered = matches_filtered[FIELDS_TO_PRINT]

    all_ban_info = matches_filtered['ban_info']
    matches_filtered['ban_info'] = build_new_ban_info_col(all_ban_info)

    jsonified = matches_filtered.to_json(orient='records')
    return jsonified

def theme_search_svd(query):
    """
    Returns a json of the most similar documents to the query.
    
    Similarity is determined using SVD similarity
    between the query and the summaries of all the books.
    """
    df = pd.read_csv("final1.csv")
    nan_variable = "summary"
    df_cleaned = df[nan_variable].fillna('')
    lst_blurb = df_cleaned

    docs_compressed_normed, words_compressed_normed, query_vec = analysis.svd_analysis(lst_blurb, query)
    # MAKE SURE TO TRANSPOSE words_compressed_normed!!!
    top_matches_lst = analysis.closest_projects_to_query(docs_compressed_normed, words_compressed_normed.T, query_vec)

    # change to JSON for printing
    matches_filtered = df.iloc[top_matches_lst]
    matches_filtered = matches_filtered[FIELDS_TO_PRINT]

    all_ban_info = matches_filtered['ban_info']
    matches_filtered['ban_info'] = build_new_ban_info_col(all_ban_info)

    jsonified = matches_filtered.to_json(orient='records')
    return jsonified

@app.route("/")
def home():
    return render_template('index.html',title="sample html")

@app.route("/titles")
def titles_search():
    text = request.args.get("title")
    return title_search(text)

@app.route("/books")
def books_search():
    text = request.args.get("title")
    return theme_search_svd(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
