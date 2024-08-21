# ZeroPOIRec : Large Language Models are Zero-shot Point-of-Interest Recommenders

> Under Review on KDD 2025 Applied Data Science Track


## 1. Overview

With the increasing emphasis on privacy preservation, there is a shift towards zero-shot recommendation approaches that prioritize individual user preferences inferred from their visit history. Recently, the ability of pre-trained large language models(LLMs) to understand human behavior has emerged as an alternative to zero-shot recommendations. Accordingly, we propose a novel zero-shot recommendation system using LLMs, called ZeroPOIRec, that involves a _profiler module_ that enables LLMs to extract individual user preferences and a _recommender module_ that facilitates the zero-shot POI recommendation performance. 

<p align="center">
<img src="figures/overall_figure_new.png " width="850"> 
</p>

Generating(Extracting) user's profile for POI recommendation faces several challenges due to the unique characteristics of location data and user behavior. Unlike other domains such as movies and news, a _visit_ includes the time of visit and geographical location, which introduce complex and diverse factors affecting subsequent visit locations. Consequently, recommendations for POIs must reflect the temporal patterns of users, varying by time, day, and season. Additionally, capturing geographic mobility related to users' residential or workplace locations and considering proximity in recommendations are necessary. Insights derived from users' visit types and behavioral patterns can often infer individual characteristics, which are closely linked to predicting subsequent visit locations. Therefore, capturing these complex and dynamic preferences is essential for POI recommendation.

<p align="center">
<img src="figures/intro_new.png " width="500"> 
</p>

## 2. Prompts

We designed two prompts for each module. A _profiler module_ enables LLMs to extract individual user preferences, and a _recommender module_ facilitates the zero-shot POI recommendation performance. 

### 2-1. Profiler Prompt

The profiler prompt consists of a system message and a user message. The system message typically specifies the persona that the model should adopt in its responses, and we provide descriptions of the input data and guidelines for the model's output through the system message. In the user message, we convey the objective of extracting user preferences that would assist in recommending the next visiting place. Then, we specify the factors that should be considered when extracting user preferences.Additionally, the user message includes a suggestion to explore informative features from a user's visit history provided with the prompt and focus on these features. Last, the user's recent visit history is provided at the end of the user message. By placing the instructions before the visit history data, the LLM can clearly understand the goal and process the data accordingly.

<p align="center">
<img src="figures/profiler.png " width="500"> 
</p>

### 2-2. Recommender Prompt

Similar to the profiler prompt, the recommender prompt includes the system message that describes the formats of the input data and the output, while the user message conveys the directives and data directly relevant to generating recommendation results. The data transmitted to the recommender prompt consists of the user preferences extracted from profiler prompt, the user's recent visit history, and a list of POI candidates. In the user message, the objective is specified in two steps: (i) selecting the top-$R$ places out of $N$ candidates ($R < N$) and (ii) ranking these $R$ places. Furthermore, the directive suggests assigning higher attention to aspects of the multiple preferences that are highly relevant to recently visited places. Last, the directive corresponding to highlighting key features is also incorporated in the recommender prompt.

<p align="center">
<img src="figures/recommender.png " width="500"> 
</p>

## 3. Datasets

We used four extensive real-world datasets, Seoul-Private(proprietary, not provided), Yelp2018, Foursquare New York City(NYC), and Foursquare Tokyo, containing semantic information regarding places. We provide the preprocessed datasets. For detailed information on preprocessing, please read our paper.

<p align="center">
<img src="figures/dataset.png " width="350"> 
</p>

## 4. Performance

ZeroPOIRec outperforms non-LLM baselines and LLM baselines. Comparisons with more baselines under various settings and the ablation study can be found in our paper.

<p align="center">
<img src="figures/performance.png " width="850"> 
</p>

## 🚀 Quick Start 🚀


1. Write your own OpenAI API keys into `api/api_key.py`
2. Intall OPENAI
    ```bash
    pip install openai==0.28
    ```
3. Evaluate ZeroPOIRec.
    ```bash
    python main.py data

    # e.g 
    # python main.py NYC
    ```


