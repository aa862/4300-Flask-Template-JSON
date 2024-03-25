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

fields_to_print = ['title','authors','ban_info', 'summary']
# Sample search using json with pandas
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
    matches_filtered = matches_filtered[fields_to_print]
    all_ban_info = matches_filtered['ban_info']
    matches_filtered['ban_info'] = edit_ban_info_print(all_ban_info)

    jsonified = matches_filtered.to_json(orient='records')
    return jsonified

def theme_search(query):
    """
    Returns a json of the most similar documents to the query.
    
    Similarity is determined using cosine similarity
    between the query and the summaries of all the books.
    """
    df = pd.read_csv("final1.csv")
    
    nan_variable = "summary"
    df_cleaned = df[nan_variable].fillna('')
    lst_blurb = df_cleaned

    cossim_results = analysis.get_doc_rankings(query, lst_blurb)
    matches_filtered = df.iloc[cossim_results]
    matches_filtered = matches_filtered[fields_to_print]
    all_ban_info = matches_filtered['ban_info']
    matches_filtered['ban_info'] = edit_ban_info_print(all_ban_info)
    jsonified = matches_filtered.to_json(orient='records')
    return jsonified

def build_ban_freq_dict(ban_data_str: str) -> dict:
    state_freq_dict = {}
    # print("ban_data_str")
    # print(ban_data_str)
    data_str = ban_data_str.strip().split(";")[:-1]
    # print("data_str")
    # print(data_str)
    for ban_instance in data_str:
        data_fields = ban_instance.split(",")
        # print("data_fields")
        # print(data_fields)
        state = data_fields[1].strip()
        if state in state_freq_dict:
            state_freq_dict[state] += 1
        else:
            state_freq_dict[state] = 1
    return state_freq_dict

def edit_ban_info_print(df_col):
    """
    Returns a list of strings to be used as the new column for printing.
    """
    # list of dictionaries of mappings of state to frequency of banning
    ban_freq_dict_lst = []
    # dictionaries of mappings of state to frequency of banning
    ban_freq_dict = {}
    # title_ban_info = None
    for ban_data_str in df_col:
        ban_freq_dict = build_ban_freq_dict(ban_data_str)
        ban_freq_dict_lst.append(ban_freq_dict)
    # list of strings will replace entries for ban info
    new_lst = []
    for lst_dict_elem in ban_freq_dict_lst:
        state_ban_info_str = ""
        for state in lst_dict_elem:
            body_str = state + " (" + str(lst_dict_elem[state]) + "), "
            state_ban_info_str = state_ban_info_str + body_str
        new_lst.append(state_ban_info_str[:-2])
    return new_lst

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
    return theme_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)




