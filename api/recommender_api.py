import openai
import pandas as pd
import json

def recommender_api(data, user_preference, user_recent_log, poi_list, model="gpt-3.5-turbo-0613", temperature = 0,  verbose=False):
    
    if data == "Yelp":
        num_features = 4
        features_list = ['index', 'place_name', 'place_category', 'place_address']

    elif data == "NYC" or data == "Tokyo" :
        num_features = 3
        features_list = ['index', 'place_category', 'place_address']
    
    with open('./api/recommender_system_prompt.txt', 'r') as file:
        system_prompt = file.read()
    system_prompt = system_prompt.format(num_features=num_features, features_list=features_list)
    
    with open('./api/recommender_user_prompt.txt', 'r') as file:
        user_prompt = file.read()
    
    user_content = f"""{user_prompt}

    ``` User Preference
    {user_preference}
    ```

    *** User's recently visited places
    {user_recent_log}
    ***

    ### List of 31 places
    {poi_list}
    ###

    """

    if verbose:
        print(user_content)

    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
                ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    output = response.choices[0].message["content"]
    token = response.usage["total_tokens"]
    # print(output)

    try:
        data_list = [json.loads(line) for line in output.split('\n') if line]
        df = pd.DataFrame(data_list)
        # print(df)
    except:
        df = output
        # print(output)

    return df, token