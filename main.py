from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import text
import json
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import imdb










with open('.env', 'r') as json_file:
    env = json.load(json_file)

db = create_engine(
    env["url"],
    connect_args={'sslmode':'require','options': '-csearch_path={}'.format(env["dbschema"])},
    echo=False,
).connect()


def title_query(n_rows,offset):
    cluster_feed_query=f"""
        WITH title_basics_l AS (SELECT * from title_basics LIMIT {int(n_rows)} OFFSET {offset})
        SELECT DISTINCT ON ("tconst") "isAdult", "startYear", "nconst", "genres", "runtimeMinutes", "language", "directors", "writers", "characters", "tconst"
        FROM title_basics_l
            LEFT JOIN title_akas
                ON "tconst" = title_akas."titleId"
            LEFT JOIN title_crew
                USING("tconst")
            LEFT JOIN title_ratings
                USING("tconst")
            LEFT JOIN title_principals
                USING("tconst")
        --WHERE "cluster" == ... ;
    """
    return cluster_feed_query

get_length="""
SELECT COUNT(DISTINCT "tconst") FROM title_basics;
"""
length=pd.read_sql(get_length,db)["count"][0]
print(length)




df=pd.DataFrame()

def fetch_movie_info(length,n_rows):
    df=pd.DataFrame()
    for n in range(int(length)//n_rows):
        df=pd.concat([df,pd.read_sql(title_query(n_rows,n*n_rows),db)])
        print(f"{round(n/(length//n_rows)*100,3)}% done")
    return df.reset_index(drop=True)


def fetch_random_movie_info(length): #migth be shorter than the specified length
    x=1000
    def query(x):
        query=f"""
        CREATE EXTENSION IF NOT EXISTS tsm_system_rows;
        CREATE OR REPLACE VIEW title_random AS 
            SELECT *
            FROM title_basics
            TABLESAMPLE SYSTEM_ROWS({x});
        SELECT DISTINCT ON ("tconst") "isAdult", "startYear", "nconst", "genres", "runtimeMinutes", "language", "directors", "writers", "characters", "tconst", "originalTitle"
            FROM title_random
                LEFT JOIN title_akas
                    ON "tconst" = title_akas."titleId"
                LEFT JOIN title_crew
                    USING("tconst")
                LEFT JOIN title_ratings
                    USING("tconst")
                LEFT JOIN title_principals
                    USING("tconst");
        """
        return query
    df=pd.DataFrame()
    for n in range(length//1000):
        df=pd.concat([df,pd.read_sql(query(x),db)])
        print(f"{round(n/(length//1000)*100,3)}% done")
    if length%1000 != 0 : 
        x=length%1000
        df=pd.concat([df,pd.read_sql(query(x),db)])
    print("100% done")
    return df.drop_duplicates().reset_index(drop=True)

#blob=fetch_movie_info(length/100,1000)
random_blob=fetch_random_movie_info(400)


#print(blob.head())
print(random_blob.shape)
print(random_blob.columns)

ordi_encoder = OrdinalEncoder()
cat_encoder = OneHotEncoder(sparse_output=True)
vectorizer=CountVectorizer()
#matrix_transformer = FunctionTransformer(lambda x : " ".join([e for e in x]))
svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
scaler = MinMaxScaler()

def matrix_transformer_function(x):
    L=[]
    for i in range(x.shape[0]):
        s=""
        for j in range(x.shape[1]):
            if x[i][j] == None : y=""
            else : y=x[i][j]
            s = s + " " + str(y)
        L.append(s)
    return L

class matrix_transformer:
    global matrix_transformer_function
    def __init__(self):
        pass
    def fit(self,x):
        return matrix_transformer_function(self)
    def fit_transform(self,x):
        return matrix_transformer_function(self)
    def transform(self):
        return matrix_transformer_function(self)


preprocessor1 = ColumnTransformer(
    transformers=[
        #("scaler",scaler,["startYear"]),
        ("select","passthrough",["genres","directors","writers","characters","nconst","language","startYear"]),
    ]
)

preprocessor=Pipeline([('preproc1', preprocessor1),('matrix_transformer',matrix_transformer),('vectorizer', vectorizer)])#,('preproc2', preprocessor2),('svd',svd)])

cluster_algo=KMeans(n_clusters=200,random_state=0,verbose=True)

#el_pipo = Pipeline([('preproc', preprocessor), ('kmeans', cluster_algo)])

x_p=preprocessor.fit_transform(random_blob)
x=cluster_algo.fit_transform(x_p)

print(cluster_algo.cluster_centers_)
