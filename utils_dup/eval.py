import numpy as np
from bs4 import BeautifulSoup

import pandas as pd
from google.colab import data_table


def _convert_HTML(text):
    # remove duplicate text + extract code blocks 
    # return pure text and code blocks
    
    bs_text = BeautifulSoup(text, features="lxml")
    text = "Possible Duplicate:"
    dup_can = bs_text.find(lambda tag: tag.name == "strong" and text in tag.text)
    if dup_can!=None:
        wrapping = dup_can.find_previous('blockquote')
        if wrapping!= None: wrapping.clear()
#         print(dup_can.parent.parent)

    text = "This is a duplicate of"
    dup_can = bs_text.find(lambda tag: tag.name == "p" and text in tag.text)
    if dup_can!=None:
        wrapping = dup_can.clear()

    # extract code block
    
    cb_list = []
#     cbs = bs_text.find_all("code")
    cbs = bs_text.select("pre > code")
    
    if len(cbs)!=0:
        for cb in cbs:
            cb_list.append(cb.get_text())
            cb.clear()
            
    text = bs_text.get_text()
    return text, cb_list


def _print_post(post):
    # print('Id:',post['Id'])
    # print('Title:',post['Title']+'\n')
    # print('Body:',_convert_HTML(post['Body'])[0])
    # print('-----------------------------------')
    # Select specific keys
    selected_keys = ['Id', 'Title', 'Body']

    filtered_post = {key: post[key] for key in selected_keys if key in post}
    df = pd.DataFrame(list(filtered_post.items()), columns=['Key', 'Value'])
    # print(df)
    data_table.DataTable(df, include_index=False)
    print('-----------------------------------')
    # data_table.DataTable(df, include_index=False)

def return_pred_top_n(i, testset, label_list, ranking_list, top_n = 3):

    list_key = list(testset['dup'].keys())
    query = testset['dup'][list_key[i]]
    print('Query:')
    _print_post(query)
    gt = testset['rel'][list_key[i]]
    
    rl = ranking_list[i].numpy().astype(int)
    for i_g, g in enumerate(gt):
        cur_gt = testset['ori'][g]

        ind_pred = np.where(rl==g)
        if len(ind_pred)!=0:
            print(f"Candidate {ind_pred[0]} is the ground truth.")
        else:
            print('The ground truth is not in the list.')
        
        print('ground truth %d:'%(i_g))
        _print_post(cur_gt)
    
    for i_n in range(top_n):
        print('Candidate %d:'%i_n)
        _print_post(testset['ori'][int(ranking_list[i][i_n].item())])
