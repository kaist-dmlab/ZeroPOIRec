import pandas as pd
from pandas import Timestamp
import numpy as np
import random
from random import shuffle

import openai
import os, sys
import json
import re
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import zeropoirec

def main():
    print("------------------------------------------------------------------------")
    print("This code is for running ZeroPOIRec.")
    print("You need your api key from OPEN AI.")
    print("------------------------------------------------------------------------")

    if len(sys.argv) != 2:
        print("Run Cmd: python main.py data")
        print("\nParamters -----------------------------------------------------------")
        print("1. data: {Yelp, NYC, Tokyo}")
        
        sys.exit(-1)

    # For user parameters
    data = sys.argv[1]
    
    if data == "Yelp":
        user_data = pd.read_csv('./datasets/preprocessed_yelp_100.csv')
        poi_data = pd.read_csv('./datasets/business_info_yelp.csv')
        poi_data['addr'] = (poi_data['address'].fillna('') + ' ' +\
                  poi_data['city'].fillna('') + ' ' +\
                  poi_data['state'].fillna('')).replace(',', '')
        poi_data['cat'] = poi_data['categories'].fillna('').apply(lambda x: x.split(',')[0])
        poi_data.rename(columns = {'business_id' : 'pid', 'name' : 'placename'}, inplace=True)
        
    elif data == "NYC":
        user_data = pd.read_csv('./datasets/preprocessed_nyc_100.csv')
        poi_data = pd.read_csv('./datasets/business_info_nyc.csv')
        poi_data.rename(columns = {'venueId' : 'pid', 'venueCategory' : 'cat', 'address' : 'addr'}, inplace=True)
        
    elif data == "Tokyo":
        user_data = pd.read_csv('./datasets/preprocessed_tky_100.csv')
        poi_data = pd.read_csv('./datasets/business_info_tky.csv')
        poi_data.rename(columns = {'venueId' : 'pid', 'venueCategory' : 'cat', 'address' : 'addr'}, inplace=True)

    # Parameters for GPT API
    # model = "gpt-3.5-turbo-0613"
    # temperature = 0
    
    # start recommend
    zeropoirec.zeropoirec(data, user_data, poi_data)

if __name__ == '__main__':
    print(sys.argv)
    main()