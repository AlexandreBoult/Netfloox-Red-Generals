import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse 

# Database URL
db_url = "postgresql://citus:floox2024!@c-groupe4.tlvz7y727exthe.postgres.cosmos.azure.com:5432/netfloox"

# Create engine
engine = create_engine(db_url)

# Define the SQL query to fetch data from the view
sql_query_rec = """SELECT * FROM netfloox_complet.recommendation"""
sql_query_pop = """SELECT * FROM netfloox_complet.features"""

# Read the data into a pandas DataFrame
dfRec = pd.read_sql(sql_query_rec, engine)
dfPop = pd.read_sql(sql_query_pop, engine)

# Close the database connection
engine.dispose()
print(f"Taille de pop avec NULL {dfPop.size}")
dfPop = dfPop.dropna()
print(f"Taille de pop finale {dfPop.size}")
print(f"Taille de rec avec NULL {dfRec.size}")
dfRec = dfRec.dropna()
print(f"Taille de pop finale {dfRec.size}")


ctpop = ColumnTransformer([
    ('num', StandardScaler(), ['startYear', 'runtimeMinutes']),
    ('genres', CountVectorizer(), 'genres'),
    ('actors',CountVectorizer(), 'actors')  #LE VECTORIZER NE FUNCTIONNE QUE AVEC UNE COLONNE

])

pop = Pipeline([
    ('prep', ctpop),
    ('model', RandomForestRegressor())
])

X = dfPop[['genres', 'startYear', 'runtimeMinutes','actors']]
y2 = dfPop ['numVotes']
y1 = dfPop['averageRating']

parameters = [{'model__criterion':['absolute_error'],
               'model__n_estimators':[100,200],
              'model__max_depth': [4,9,15], 
              'model__max_features': [1,5,10],               
              'model__min_samples_split': [2,4],
              'model__min_samples_leaf': [1,3]}]

clf = GridSearchCV(pop, parameters, scoring = "neg_mean_absolute_error")

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.33, random_state=42)
# Combine y1_train and y2_train into a 2D array
y_train_combined = np.column_stack((y1_train, y2_train))

pop.fit(X_train, y_train_combined)

# Make predictions on the testing data
y_pred_combined = pop.predict(X_test) 

# Split the predictions into separate arrays for y1 and y2
y1_pred = y_pred_combined[:, 0]
y2_pred = y_pred_combined[:, 1]

# Evaluate the model
mse_y1 = mean_squared_error(y1_test, y1_pred)
mse_y2 = mean_squared_error(y2_test, y2_pred)
print("Score test Rating:", mse_y1)
print("Score test Votes:", mse_y2)

features = ['genres','actors','directors']  

def combine_features(row):     
    return row['genres'] +" "+row['actors']+" "+row["directors"]

for feature in features:    
     dfRec[feature] = dfRec[feature].fillna('') 
     dfRec["combined_features"] = dfRec.apply(combine_features,axis=1) 
cv = CountVectorizer() 
count_matrix = cv.fit_transform(dfRec["combined_features"]) 
cosine_sim = cosine_similarity(count_matrix)  
     
def get_title_from_index(index):     
    return dfRec[dfRec.index == index]["primaryTitle"].values[0] 

def get_index_from_title(title):     
    return dfRec[dfRec['primaryTitle'] == title].index[0]

#movie_user_likes = "Avatar" 
#movie_index = get_index_from_title(movie_user_likes) 
similar_movies =  list(enumerate(cosine_sim[5]))  
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

print(pd.DataFrame(sorted_similar_movies).head(5))