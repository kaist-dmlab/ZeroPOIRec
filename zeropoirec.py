import pandas as pd
from pandas import Timestamp
import numpy as np
import random
from random import shuffle
import csv

import openai
import os, sys
import json
import re

from api.profiler_api import *
from api.recommender_api import *
from api.api_key import *


from tqdm import tqdm


def make_user_log(data, user_data, uid, poi_data, cnt):
    
    if data == "Yelp":
        user_log = pd.DataFrame()
        user_log['date'] = eval(user_data[user_data['hash_uid'] == uid]['visited_dates'][cnt])
        user_log['pid'] = eval(user_data[user_data['hash_uid'] == uid]['visited_pids'][cnt])
        place_name = []
        place_category = []
        place_address = []
        for j in user_log['pid']:
            place_name.append(poi_data[poi_data['pid'] == j]['placename'].fillna('').values[0])
            place_category.append(poi_data[poi_data['pid'] == j]['cat'].fillna('').values[0])
            address = poi_data[poi_data['pid'] == j]['addr'].fillna('').values[0]
            place_address.append(address.replace(",", ""))
        user_log['place_name'] = place_name
        user_log['place_category'] = place_category
        user_log['place_address'] = place_address
        user_log = user_log.drop(['pid'], axis=1)
        
    elif data == "NYC" or data == "Tokyo" :
        user_log = pd.DataFrame()
        user_log['date time'] = eval(user_data[user_data['hash_uid'] == uid]['visited_times'][cnt])
        user_log['pid'] = eval(user_data[user_data['hash_uid'] == uid]['visited_pids'][cnt])
        place_category = []
        place_address = []
        for j in user_log['pid']:
            place_category.append(poi_data[poi_data['pid'] == j]['cat'].fillna('').values[0])
            address = poi_data[poi_data['pid'] == j]['addr'].fillna('').values[0]
            place_address.append(address.replace(",", ""))
        user_log['place_category'] = place_category
        user_log['place_address'] = place_address
        user_log = user_log.drop(['pid'], axis=1)
        user_log['date'] = user_log['date time'].apply(lambda x: x.date())
        user_log['time'] = user_log['date time'].apply(lambda x: x.time())
        user_log = user_log.drop(['date time'], axis=1)
        user_log = user_log[['date', 'time', 'place_category', 'place_address']]
        
    return user_log

def make_candidates_list(data, user_data, uid, poi_data, cnt):
    
    candidates_pid = eval(user_data[user_data['hash_uid'] == uid]['candidates'][cnt])
    target_pid = user_data[user_data['hash_uid'] == uid]['test_pid'][cnt]
    candidates_pid.append(target_pid)
    
    
    if data == "Yelp":
        candidates_list = pd.DataFrame()
        place_name = []
        place_category = []
        place_address = []
        for j in candidates_pid:
            place_name.append(poi_data[poi_data['pid'] == j]['placename'].fillna('').values[0])
            place_category.append(poi_data[poi_data['pid'] == j]['cat'].fillna('').values[0])
            address = poi_data[poi_data['pid'] == j]['addr'].fillna('').values[0]
            place_address.append(address.replace(",", ""))
        candidates_list['place_name'] = place_name
        candidates_list['place_category'] = place_category
        candidates_list['place_address'] = place_address
        candidates_list['pid'] = candidates_pid
        candidates_list.drop_duplicates(inplace=True)
        target_idx = candidates_list.index[candidates_list["pid"] == target_pid][0]
        candidates_list['final_score'] = [0]*len(candidates_list)
    
    elif data == "NYC" or data == "Tokyo" :
        candidates_list = pd.DataFrame()
        place_category = []
        place_address = []
        for j in candidates_pid:
            place_category.append(poi_data[poi_data['pid'] == j]['cat'].fillna('').values[0])
            address = poi_data[poi_data['pid'] == j]['addr'].fillna('').values[0]
            place_address.append(address.replace(",", ""))
        candidates_list['place_category'] = place_category
        candidates_list['place_address'] = place_address
        candidates_list['pid'] = candidates_pid
        candidates_list.drop_duplicates(inplace=True)
        target_idx = candidates_list.index[candidates_list["pid"] == target_pid][0]
        candidates_list['final_score'] = [0]*len(candidates_list)
        
    
    return candidates_list, target_idx

def tab_separated_format_user_log(data, user_log):
    
    if data == "Yelp":
        user_log_str = user_log.to_string()
        
        user_log_latest = user_log[-5:]
        user_log_latest_str = user_log_latest.to_string()
        
    elif data == "NYC" or data == "Tokyo" :
        user_log_str = user_log.to_string(col_space=[20, 20, 30, 30])
        
        user_log_latest = user_log[-5:]
        user_log_latest_str = user_log_latest.to_string(col_space=[20, 20, 30, 30])
    
    user_log_tab = 'index' + re.sub('  +', '\t', user_log_str)
    user_log_tab = re.sub('date hour', 'date\thour', user_log_tab)
    
    user_log_latest_tab = 'index' + re.sub('  +', '\t', user_log_latest_str)
    user_log_latest_tab = re.sub('date hour', 'date\thour', user_log_latest_tab)
    
    return user_log_tab, user_log_latest_tab

def tab_separated_format_candidates_list(data, candidates_list):
    
    if data == "Yelp":
        candidates_list = candidates_list[['place_name', 'place_category', 'place_address']]
        candidates_list_str = candidates_list.to_string()
        candidates_list_tab = 'index' + re.sub('  +', '\t', candidates_list_str)
        
    elif data == "NYC" or data == "Tokyo" :
        candidates_list = candidates_list[['place_category', 'place_address']]
        candidates_list_str = candidates_list.to_string(col_space=[30, 30])
        candidates_list_tab = 'index' + re.sub('  +', '\t', candidates_list_str)
    
    
    return candidates_list_tab
        

def zeropoirec(data, user_data, poi_data):
    print('Start recommendation using ZeroPOIRec!')
    
    print('api_key : ', openai.api_key)
    
    uid_list = user_data['hash_uid']
    
    cnt = 0
    k_list = []
    
    for i in tqdm(uid_list[:]):
        
        # make user's log and candidates list
        user_log = make_user_log(data, user_data, i, poi_data, cnt)
        candidates_list, target_idx = make_candidates_list(data, user_data, i, poi_data, cnt)
        
        # data formatting
        user_log_tab, user_log_latest_tab = tab_separated_format_user_log(data, user_log)
              
        # extraction preference
        preference_output, _ = profiler_api(data, user_log_tab)
        
        # ensembling for consistency
        # change the order of candidates list
        for r in range(3):
            candidates_list_r = candidates_list.sample(frac=1).reset_index(drop=True)
            candidates_list_tab = tab_separated_format_candidates_list(data, candidates_list_r)
            
            res, _ = recommender_api(data, preference_output, user_log_latest_tab, candidates_list_tab)
            res[f'r{r}_score'] = len(candidates_list) + 1 - res['rank']
            res = res.rename(columns={'index': f'r{r}_idx'})
            
            candidates_list_r_idx = candidates_list_r.reset_index().rename(columns={'index': f'r{r}_idx'})
            candidates_list = pd.merge(candidates_list, candidates_list_r_idx, how='left')
            candidates_list = pd.merge(candidates_list, res[[f'r{r}_idx', f'r{r}_score']], on=f'r{r}_idx', how='left').fillna(0)
            candidates_list['final_score'] += candidates_list[f'r{r}_score']
            
        
        # save the position of target place
        candidates_list_sorted = candidates_list.sort_values(by='final_score', ascending=False).reset_index()
        forecast_target = candidates_list_sorted[candidates_list_sorted['index'] == target_idx].index[0]
        k_list.append(forecast_target)
        
        cnt += 1
        
    print('end recommendation!')    
    k_list_arr = np.array(k_list)

    for i in [1,2,3,5]:
        print('hit rate@',i)
        print(sum(k_list_arr < i) / len(k_list_arr))
            
            
            
            