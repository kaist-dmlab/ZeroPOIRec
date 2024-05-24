import openai
    
def profiler_api(data, user_log, model= "gpt-3.5-turbo-0613", temperature = 0, verbose=False):
    
    if data == "Yelp":
        num_features = 5
        features_list = ['index', 'date', 'place_name', 'place_category', 'place_address']

    elif data == "NYC" or data == "Tokyo" :
        num_features = 5
        features_list = ['index', 'date', 'time', 'place_category', 'place_address']
    
    with open('./api/profiler_system_prompt.txt', 'r') as file:
        system_prompt = file.read()
    system_prompt = system_prompt.format(num_features=num_features, features_list=features_list)
    
    with open('./api/profiler_user_prompt.txt', 'r') as file:
        user_prompt = file.read()

    
    user_content = f"""{user_prompt}

    ```
    {user_log}
    ```

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

    return output, token