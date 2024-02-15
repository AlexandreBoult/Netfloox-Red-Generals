import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
import numpy as np

# Database URL
db_url = "postgresql://citus:floox2024!@c-groupe4.tlvz7y727exthe.postgres.cosmos.azure.com:5432/netfloox"

# Create engine
engine = create_engine(db_url)

# Define the SQL query to fetch data from the view
sql_query = """SELECT * FROM netfloox_complet.features"""

# Read the data into a pandas DataFrame
dfPop = pd.read_sql(sql_query, engine)

# Close the database connection
engine.dispose()
print(dfPop.size)
dfPop = dfPop.dropna()
print(dfPop.size)

ct = ColumnTransformer([
    ('num', StandardScaler(), ['startYear', 'runtimeMinutes']),
    ('genres', CountVectorizer(), 'genres'),
    ('actors', CountVectorizer(), 'actors'),
    ('directors', CountVectorizer(), 'directors') #LE VECTORIZER NE FUNCTIONNE QUE AVEC UNE COLONNE

])

pipe = Pipeline([
    ('prep', ct),
    ('model', RandomForestRegressor(criterion = 'absolute_error', max_depth=4, max_features=0.3, min_samples_split=3))
])

X = dfPop[['genres', 'startYear', 'runtimeMinutes', 'actors', 'directors']]
y2 = dfPop ['numVotes']
y1 = dfPop['averageRating']

""" parameters = [{'model__criterion':['absolute_error'],
               'model__n_estimators':[100,200],
              'model__max_depth': [4], 
              'model__max_features': [0.3],               
              'model__min_samples_split': [3]}]

clf = GridSearchCV(pipe, parameters, scoring = "neg_mean_absolute_error") """

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.33, random_state=42)
# Combine y1_train and y2_train into a 2D array
y_train_combined = np.column_stack((y1_train, y2_train))

pipe.fit(X_train, y_train_combined)

# Make predictions on the testing data
y_pred_combined = pipe.predict(X_test) 

# Split the predictions into separate arrays for y1 and y2
y1_pred = y_pred_combined[:, 0]
y2_pred = y_pred_combined[:, 1]

# Evaluate the model
mse_y1 = mean_squared_error(y1_test, y1_pred)
mse_y2 = mean_squared_error(y2_test, y2_pred)
print("Score test Rating:", mse_y1)
print("Score test Votes:", mse_y2)