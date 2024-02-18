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
dfPop = dfPop.dropna()
dfRec = dfRec.dropna()

class CommaSeparatedEncoder:
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoder = OneHotEncoder()
        
    def fit(self, X, y=None):
        for col in X.columns:
            labels = set([val.strip() for sublist in X[col].str.split(',') for val in sublist])
            self.label_encoders[col] = LabelEncoder().fit(list(labels))
        return self
    
    def transform(self, X):
        encoded_features = []
        for col in X.columns:
            encoded_col = self.label_encoders[col].transform(X[col].apply(lambda x: [val.strip() for val in x.split(',')]))
            encoded_features.append(encoded_col)
        encoded_features = np.column_stack(encoded_features)
        return self.one_hot_encoder.fit_transform(encoded_features)

ctpop = ColumnTransformer([
    ('num', StandardScaler(), ['startYear', 'runtimeMinutes']),
    ('genres', CommaSeparatedEncoder(), ['genres','actors']) #LE VECTORIZER NE FUNCTIONNE QUE AVEC UNE COLONNE

])

ctrec = ColumnTransformer([
    ('num', StandardScaler(), ['startYear', 'averageRating', 'numVotes']),
    ('genres', CommaSeparatedEncoder(), ['genres','actors','directors']),

])

pop = Pipeline([
    ('prep', ctpop),
    ('model', RandomForestRegressor())
])

recom = Pipeline([
    ('prep', ctrec),
    ('similarity', FunctionTransformer(lambda X: cosine_similarity(X)))
])

X = dfPop[['genres', 'startYear', 'runtimeMinutes']]
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


Xrec = dfRec.drop(columns=['primaryTitle'])

recom.fit(Xrec)

def get_recommendations(target_title, cosine_sim_matrix, dfRec, top_n=5):
    # Find the index of the target_tconst in dfRec
    target_index = dfRec[dfRec['primaryTitle'] == target_title].index[0]

    # Sort the cosine similarity values for the target sample
    similar_indices = np.argsort(-cosine_sim_matrix[target_index])

    # Select the top N similar samples (excluding itself)
    top_n_similar_indices = similar_indices[1:top_n+1]  # Exclude itself

    # Provide recommendations from the 'tconst' column
    recommendations = dfRec.iloc[top_n_similar_indices]['primartTitle']
    return recommendations.values

# Specify the target tconst for which you want recommendations
target_title = 'The Godfather'

# Get recommendations for the target tconst
recommendations = get_recommendations(target_title, cosine_sim_matrix, dfRec)
print("Recommendations based on cosine similarity for tconst", target_tconst, ":")
print(recommendations)