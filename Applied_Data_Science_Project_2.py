#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from dash_table import DataTable 


# # **About the dataset**   
# **The MovieLens datasets describe ratings and free-text tagging activities from MovieLens, a movie recommendation service. The full dataset contains nearly 20M ratings and 465000 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015.**    
# 
# **The datasets I use in this project are:**   
# **a. movies_metadata.csv that contains information about movie_id, title, genres, etc.**  
# **b. credits.csv that contains information about the cast and crew of a particular movie.**   
# **c. keywords.csv that contains information about the sub-genres of a movie.**   
# **d. ratings.csv that contains information about the ratings given to a movie by different users.**    

# In[3]:


#reading the movies_metadata dataset
movies_metadata = pd.read_csv('https://mrssamardeep.s3.ca-central-1.amazonaws.com/movies_metadata.csv', encoding = 'utf-8', error_bad_lines= 'coerce', engine = 'python')


# In[4]:


movies_metadata.head()


# In[5]:


movies_metadata.shape #The dataset consists of 45466 rows and 24 columns


# In[6]:


movies_metadata.columns


# ## **Simple Recommender(IMDB 250 Clone)**

# **'v' is the number of votes garnered by the movie**  
# **'m' is the minimum number of votes required for the movie to be in the chart (the prerequisite)**  
# **'R' is the mean rating of the movie**  
# **'C' is the mean rating of all the movies in the dataset**    
# 
# ### **The formula used for generating the weighted ratings is:**   
# $$(\frac{v}{v+m}* R) + (\frac{m}{v+m} * C)$$  
# 

# In[7]:


m = movies_metadata['vote_count'].quantile(0.85) #We set 'm' to be greater than the 85th percentile of number of votes receicved
m


# In[8]:


#Selecting only the movies that have a runtime between 45 minutes and 300 minutes. Also, these movies have atleast 82 votes.
#Qualified movies
q_movies = movies_metadata.loc[(movies_metadata.runtime >= 45) & (movies_metadata.runtime <= 300)]
q_movies = q_movies.loc[q_movies['vote_count'] > m]
q_movies.shape


# In[9]:


C = movies_metadata['vote_average'].mean() #The mean rating of all movies in the dataset
C


# In[10]:


#Function to compute the IMDB weighted rating for each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    score = (v/(v+m) * R) + (m/(v+m) * C)
    return score


# In[11]:


#Computing the score based on weighted function defined above
q_movies['Score'] = q_movies.apply(weighted_rating, axis = 1)


# In[12]:


q_movies = q_movies.sort_values('Score', ascending = False) #Sorting the values based on score in descending order
q_movies[['title', 'vote_count', 'vote_average', 'runtime', 'Score']].head(25)


# ## **Content Based Recommender**

# **The recommender uses the following metadata to generate recommendations**   
# **1. Movie Genre**  
# **2. Director of the movie**   
# **3. Actors (top 5)**   
# **4. Keywords/Sub-genres (top 5)**

# In[13]:


content_df = movies_metadata[['id','title', 'genres']] #collecting the relevant columns in a new dataframe


# In[14]:


content_df.head()


# In[15]:


content_df.dtypes


# In[16]:


#Filling all the NaN values in 'genres' with empty list and converting it to list object using literal_eval 
from ast import literal_eval
content_df['genres'] = content_df['genres'].fillna('[]')
content_df['genres'] = content_df['genres'].apply(literal_eval)


# In[17]:


#The 'genres' feature consists of list of dictionary
#Extracting the name of genre from the list of dictionary
content_df['genres'] = content_df['genres'].apply(lambda x: [i['name'] for i in x])
content_df.head()


# In[18]:


#Reading the credits and keywords datasets to extract the names of actors, director and sub-genres
cred_df = pd.read_csv('https://mrssamardeep.s3.ca-central-1.amazonaws.com/credits.csv', encoding = 'utf-8', engine = 'python', error_bad_lines=False)
key_df = pd.read_csv('https://mrssamardeep.s3.ca-central-1.amazonaws.com/keywords.csv', encoding = 'utf-8', engine = 'python', error_bad_lines = False)


# In[19]:


cred_df.head()


# In[20]:


key_df.head()


# In[21]:


#A function to convert 'id' values to integer 
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan


# In[22]:


content_df['id'] = content_df['id'].apply(clean_ids) #Converting the dtype of 'id' in content_df to integer
content_df.head()


# In[23]:


#Filtering out all the rows from the content_df that have null values for 'id'
content_df = content_df[content_df['id'].notnull()]
content_df.head()


# In[24]:


key_df['id'] = key_df['id'].astype('int')
cred_df['id'] = cred_df['id'].astype('int')
content_df['id'] = content_df['id'].astype('int')


# In[25]:


#Merging the keywords and credits datasets to content_df
content_df = content_df.merge(key_df, on = 'id')
content_df = content_df.merge(cred_df, on = 'id')
content_df.head()


# In[26]:


#Converting 'keywords', 'cast' and 'crew' into list objects using literal eval
features = ['keywords', 'cast', 'crew']
for feature in features:
    content_df[feature] = content_df[feature].apply(literal_eval)


# In[27]:


content_df.iloc[0]['crew'] #Each value in the 'crew' feature is a list of dictionary


# In[28]:


#Extracting director's name from crew members
def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan


# In[29]:


content_df['director'] = content_df['crew'].apply(get_director)
content_df.head()


# In[30]:


#Function to extract top 5 values from a list 
def top_five(x):
    names = [i['name'] for i in x]
    if len(names) > 5:
        return names[:5]
    return names


# In[31]:


#Extracting first 5 cast members 
content_df['cast'] = content_df['cast'].apply(top_five)
content_df.head()


# In[32]:


#Extracting first 5 keywords for a movie
content_df['keywords'] = content_df['keywords'].apply(top_five)
content_df.head()


# In[33]:


#Function to remove spaces between words and converting into lowercase so that they can be used with count vectorizer
def sanitize(x):
  if isinstance(x,list):
    return [str.lower(i.replace(' ', '')) for i in x]
  else:
    if isinstance(x, str):
      return str.lower(x.replace(' ', ''))
    else:
      return ''


# In[34]:


for feature in ['cast', 'director', 'genres', 'keywords']:
  content_df[feature] = content_df[feature].apply(sanitize)


# In[35]:


def mixture(x):
  return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])


# In[36]:


content_df['metadata'] = content_df.apply(mixture, axis = 1) #Joining the features 'keywords, 'cast', 'director' and 'genres' to form a fina feature named 'metadata'


# In[37]:


#Considering only 15000 rows from the dataset because of lack of computational power 
new_content_df = content_df.iloc[:15000, :]


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words = 'english') # Initializing the CountVectorizer
count_matrix = vec.fit_transform(new_content_df['metadata']) # Applying the bag of words technique to convert the text in 'metadata' feature to vector


# In[39]:


count_matrix.shape


# In[40]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(count_matrix, count_matrix) #Using cosine similarity to find the similarity between each row in the 'metadata' column


# In[41]:


#Storing index of each movie for reverse mapping
indices = pd.Series(new_content_df.index, index = new_content_df['title'])
indices


# In[42]:


def content_recommender(title, df = new_content_df, cosine_sim = similarity, indices = indices):
  idx = indices[title] #obtain the index of movie with the given title
  sim_score = list(enumerate(cosine_sim[idx])) #converting pairwise similarity scores of the given movie with other movies into list of tuples
  sim_score = sorted(sim_score, key = lambda x: x[1], reverse = True) #Sorting similarity scores in descending order
  sim_score = sim_score[1:11] # taking the top 10 most similar movies
  movie_indices = [i[0] for i in sim_score] # Storing the index of top 10 most similar movies
  return pd.DataFrame(df['title'].iloc[movie_indices])


# In[43]:


result_final = content_recommender('The Shawshank Redemption')
result_final


# ## **Collaborative Filtering Recommendor**

# **The technique used here is known as User Based Collaborative Filtering. It works by creating an user-item matrix where eah row represents a unique user and each column represents a unique movie. The values in the matrix represent the ratings given to a movie by a user.**    
# **Collaborative filtering makes use of the user-item matrix to find k-similar users to a particular user and recommends ratings the target user would give based on the ratings of those 'k' users.**   

# In[44]:


#Reading the rating.csv dataset
rating = pd.read_csv('https://mrssamardeep.s3.ca-central-1.amazonaws.com/ratings_small.csv', encoding = 'utf-8', error_bad_lines=False)
rating.head()


# In[45]:


#Dropping the timestamp columns as it won't help with the recommendations
rating = rating.drop('timestamp', axis = 1)
rating.head()


# In[46]:


for index, rows in rating.iterrows():
  rating.loc[index,'email'] =  str(rows[0]) + "@gmail.com"


# In[47]:


rating.isnull().sum() #There are no null values


# In[48]:


rating.shape


# ### **User-Item Matrix**

# 
# **The value in the ith row and jth column represents the rating given by user 'i' to movie 'j'.**

# In[49]:


user_item_matrix = rating.pivot_table(index = 'userId', columns = 'movieId', values = 'rating') #Creating the user-item matrix using the pivot_table function in pandas
user_item_matrix.head()


# In[50]:


user_item_matrix.shape #There are 671 users and 9066 movies


# In[51]:


#Since the user-item matrix is a sparse matrix, it consists of many null values. We can replace te null values by 0 to signify that user has not rated a movie
user_item_matrix = user_item_matrix.fillna(0) 
user_item_matrix = user_item_matrix.reset_index()
user_item_matrix.head()


# In[52]:


user_item_matrix.shape[0]
cosine_similarity([user_item_matrix[500]], [user_item_matrix[100]])


# In[53]:


rating.loc[rating['email']=='20.0@gmail.com', 'userId'].unique()[0]


# In[54]:


#Function to find similar users
def similar_users(rating, email, user_item_matrix):
  user_id = rating.loc[rating['email']==email, 'userId'].unique()[0]
  similarity = []
  for user in range(0,user_item_matrix.shape[0]):
    sim = cosine_similarity([user_item_matrix.loc[user]], [user_item_matrix.loc[int(user_id)]]) #Calculating the similarity score for each user with every other user
    similarity.append((user,sim)) #Storing the similarity score as a tuple of user_id and similarity score

  similarity = sorted(similarity, key = lambda x: x[1], reverse = True)

  most_similar_user = [tup[0] for tup in similarity]
    
  similarity_score = [tup[1] for tup in similarity]
    
  most_similar_user.remove(user_id) #Removing the original user's user_id as it will give a similarity score of 1 
    
  similarity_score.remove(similarity_score[0])
    
  return most_similar_user, similarity_score


# In[55]:


def cf_recommendation(email, rating = rating, user_item_matrix = user_item_matrix, num_movies = 10 ):
    user_id = rating.loc[rating['email']==email, 'userId'].unique()[0]
    most_similar_users = similar_users(rating, email, user_item_matrix)[0] #all the similar users to a user_id sorted in descending order
    movies_already_watched = set(list(user_item_matrix.columns[np.where(user_item_matrix.loc[user_id] > 0)])) #storing in a set all the movies already rated by the target user
    recommendations = []
    already_watched = movies_already_watched.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_movies:
            similar_user_movie_id = set(list(user_item_matrix.columns[np.where(user_item_matrix.loc[similar_user] > 0)]))
            recommendations.extend(similar_user_movie_id.difference(already_watched))#Using set diiference to find the movies that have not been rated by the target user but have been rated by similar users
            already_watched = already_watched.union(similar_user_movie_id)
        else:
            break 
    lst = []
    for ele in recommendations[:num_movies]:
        a = movies_metadata.loc[movies_metadata['id']==str(ele), 'title']
        if a.empty == False:
            lst.append(movies_metadata['title'].loc[movies_metadata['id']==str(ele)].values)
#             print(rec_df)
    series = pd.Series(lst)
    index = [i for i in range(len(series))]
    rec_df = pd.DataFrame(series,index=index,columns=['Movies'])
    return rec_df
#     return series


# In[56]:


email_list = rating['email'].unique()
movies_list = movies_metadata['title'].unique()


# In[57]:


app = dash.Dash(__name__)
dash_server = app.server
app.config["suppress_callback_exceptions"] = True
CONTENT_STYLE = {
    "margin-left": "8rem",
    "margin-right": "8rem",
    "padding": "2rem 1rem",
}
app.layout = html.Div([
    html.H1('Movie Recommendation System'),
    dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
        dcc.Tab(id = "tab1",label='Content Based Recommendation', value='tab-1'),
        dcc.Tab(id = "tab2",label='Collaborative Filtering Recommendation', value='tab-2'),
        dcc.Tab(id = "tab3", label="About", value="tab-3")
    ], colors={
        "border": "white",
        "primary": "gold",
        "background": "cornsilk"
    }),
    html.Div(id='tabs-content-props')
],style = CONTENT_STYLE)


tab1 = html.Div([
        html.Br(),
        html.H2('Please enter a movie name'),
        dcc.Input(id='username1', value='Jumanji', type='text', style={'backgroundColor':'cornsilk', 'width':'25%', 'border':'1.5px black solid', 'height': '50px'}),
        html.Br(),
        html.Br(),
        html.Button("Submit", style={'backgroundColor': 'green', 'color':'white','width':'25%', 'border':'1.5px black solid', 'height': '50px'},id ="submit-button1"),
        html.Br(),
        html.Div(id='output_div1'),
        ])

tab2 = html.Div([
        html.Br(),
        html.H2('Enter your email address'),
        dcc.Input(id='username2', value = '1.0@gmail.com', type='text', style={'backgroundColor':'cornsilk', 'width':'25%', 'border':'1.5px black solid', 'height': '50px'}),
        html.Br(),
        html.Br(),
        html.Button("Submit", style={'backgroundColor': 'green', 'color':'white','width':'25%', 'border':'1.5px black solid', 'height': '50px'}, className="me-1",id ="submit-button2"),
        html.Br(),
        html.Div(id='output_div2') ,
        ])

tab3 = html.Div([
    html.Div([
        html.H1('About the dataset'),
        html.Br(),
        html.P('The files used in this project were collected and made available by Rounak Banik, an ECE graduate from Indian Institute of Technoogy, Roorkee and a Young India Fellow. The data has been obtained from two sources: The Movie Database (TMDB) and MovieLens.  Data was collected from the MovieLens website and through a script that queried for data from various TMDB Endpoints.'),
        html.P('The following files were used in the project:'),
        html.Ul(),
        html.Li('movies_metadata.csv: The file containing metadata collected from TMDB for over 45000 movies. Data includes budget, revenue, genres, etc'),
        html.Li('credits.csv: Complete information on credits of a particular movie. Data includes director, producer, actor, etc.'),
        html.Li("ratings_small.csv: The MovieLens dataset containing 100,000 ratings on 9000 movies from 700 users."),
        html.Li('links_small.csv: Contains the list of movies that are included in the small subset of the full MovieLens dataset.'),
        html.Li('keywords.csv: Certain plot keywords associated with a particular movie'),
        html.Ul()
    ]),
    html.Div([
        html.H1('Purpose'),
        html.P('Recommendation Systems are an important application to acknowledge because use cases of recommendation systems are presented in many different real-world domains. In the scope of recommendation systems, this project tries to explore techniques such as collaborative filtering and content-based filtering that are necessary for building them. Moreover, the web application implements both the aforementioned techniques to recommend movies to users in real time.')
    ]),
    html.Div([
        html.H1('Machine Learning Tasks'),
        html.H2('Data Wrangling/Preparation'),
        html.Ul(),
        html.Li('Combining data from multiple files such as movies_metadata.csv, credits.csv and keywords.csv to form a final dataframe with relevant features.'),
        html.Li('Converting Stringified JSON objects to Python dictionaries.'),
        html.Li('Exploding the dataframe to extract top 3 values for features such as genres and cast for each movie.'),
        html.Ul(),
        html.H2('Content-Based Filtering'),
        html.Ul(),
        html.Li('Using text-preprocessing techniques such as Bag of Words to convert movies metadata into vectors.'),
        html.Li('Applying cosine similarity to generate similarities between vectors.'),
        html.Li('Defining a function to generate the top 10 movie recommendations based on cosine similarity.'),
        html.Ul(),
        html.H2('Collaborative Filtering'),
        html.Ul(),
        html.Li('Creating a user-item matrix which contains the rating that a user gives to a movie.'),
        html.Li('Defining a function based on cosine similarity to identify similar users.'),
        html.Li('Defining a recommendation function to generate movies for a user based on similar users.'),
        html.Ul(),
        html.A("Here is the link to the dataset", href='https://www.kaggle.com/rounakbanik/movie-recommender-systems/data', target="_blank")
    ])
])

@app.callback(Output('tabs-content-props', 'children'),
              Input('tabs-styled-with-props', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return tab1
    elif tab == 'tab-2':
        return tab2
    elif tab == 'tab-3':
        return tab3

@app.callback(dash.dependencies.Output('output_div1','children'),
                  [dash.dependencies.Input('submit-button1', 'n_clicks')],
                  [dash.dependencies.State('username1', 'value')],
                  )
def update_output(clicks, input1):
    if input1 not in movies_list:
        return 'Sorry, there is no movie with this name in the data. Please enter a valid name.'
    else:
        result_final = content_recommender(input1)
        dt_col_param = []
        
        if clicks is not None:
            
            for col in result_final.columns:
                dt_col_param.append({"name": str(col), "id": str(col)})
            return DataTable(
            columns=dt_col_param,
            data= result_final.to_dict('records'),
                style_cell={'textAlign': 'left'},
                                style_data={
        'color': 'black',
        'backgroundColor': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }
    ],
    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    }
        )
        else:
            html.Div(html.H4(' daaldo bhai'))

@app.callback(dash.dependencies.Output('output_div2','children'),
                  [dash.dependencies.Input('submit-button2', 'n_clicks')],
                  [dash.dependencies.State('username2', 'value')],
                  )

def update_output2(clicks, input2):
    if input2 not in email_list:
        return "Sorry, this email is not present in the data. Hence, no similar users can be found."
    else:
        result_final_cf = cf_recommendation(input2)
        print(result_final_cf)
        dt_col_param = []
        
        if clicks is not None:
            
            for col in result_final_cf.columns:
                dt_col_param.append({"name": str(col), "id": str(col)})
            return DataTable(
            columns=dt_col_param,
            data= result_final_cf.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_data={
        'color': 'black',
        'backgroundColor': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }
    ],
    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    }
        )
if __name__ == '__main__':
    app.run_server(debug=True)

