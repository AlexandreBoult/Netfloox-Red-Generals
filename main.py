from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
import pandas as pd
from sqlalchemy import text, MetaData, Table
import psycopg2
import json
import asyncio
import asyncpg
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import imdb
import pickle
import datetime
import base64
from urllib.parse import urlparse




with open('.env', 'r') as json_file:
    env = json.load(json_file)
    url = urlparse(env["url"])

host = url.hostname
port = url.port
database = url.path[1:]
user = url.username
password = url.password
"""
db = create_engine(
    env["url"],
    curect_args={'sslmode':'require','options': f'-csearch_path={env["dbschema"]}'},
    echo=True,
)
"""
conn= psycopg2.connect(
    host=host,
    port=port,
    database=database,
    user=user,
    password=password,
    options=f'-csearch_path={env["dbschema"]}'
)
cur = conn.cursor()


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

def get_title_basics_length():
    get_length="""
    SELECT COUNT(DISTINCT "tconst") FROM title_basics;
    """
    length=pd.read_sql(get_length,conn)["count"][0]
    return length



def fetch_movie_info(length,n_rows):
    df=pd.DataFrame()
    for n in range(int(length)//n_rows):
        df=pd.concat([df,pd.read_sql(title_query(n_rows,n*n_rows),conn)])
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
        df=pd.concat([df,pd.read_sql(query(x),conn)])
        print(f"{round(n/(length//1000)*100,3)}% done")
    if length%1000 != 0 : 
        x=length%1000
        df=pd.concat([df,pd.read_sql(query(x),conn)])
    print("100% done")
    return df.drop_duplicates().reset_index(drop=True)

#blob=fetch_movie_info(length/100,1000)
#random_blob=fetch_random_movie_info(400)


#print(blob.head())
#print(random_blob.shape)
#print(random_blob.columns)


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



def train_cluster(random_blob):

    ordi_encoder = OrdinalEncoder()
    cat_encoder = OneHotEncoder(sparse_output=True)
    vectorizer=CountVectorizer()
    #matrix_transformer = FunctionTransformer(lambda x : " ".join([e for e in x]))
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    scaler = MinMaxScaler()


    preprocessor1 = ColumnTransformer(
        transformers=[
            #("scaler",scaler,["startYear"]),
            ("select","passthrough",["genres","directors","writers","characters","nconst","language","startYear"]),
        ]
    )

    preprocessor=Pipeline([('preproc1', preprocessor1),('matrix_transformer',matrix_transformer),('vectorizer', vectorizer)])#,('preproc2', preprocessor2),('svd',svd)])

    cluster_algo=KMeans(n_clusters=200,random_state=0,verbose=True)

    #el_pipo = Pipeline([('preproc', preprocessor), ('kmeans', cluster_algo)])
    p=preprocessor.fit(random_blob)
    x_p=preprocessor.transform(random_blob)
    preproc,model=p,cluster_algo.fit(x_p)

    return preproc,model


def save_model(preproc,model):
    smodel=pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    spreproc=pickle.dumps(preproc, protocol=pickle.HIGHEST_PROTOCOL)
    #print(smodel)
    now = datetime.datetime.now()
    time=now.time()
    save_model_query = f"""CREATE TABLE IF NOT EXISTS model_table (time TEXT, model BYTEA, preproc BYTEA);"""
    statement=f"""INSERT INTO model_table("time", "model", "preproc") VALUES ({"'"+str(time)+"'"}, {str(base64.b64encode(smodel))[1:]}, {str(base64.b64encode(spreproc))[1:]});"""
    #print(statement)
    cur.execute(save_model_query)
    cur.execute(statement)
    print(str(time))
    return str(time)


def load_model(time):
    load_model_query = f"""SELECT "model" FROM model_table
    WHERE "time"={"'"+time+"'"};"""

    load_preproc_query = f"""SELECT "preproc" FROM model_table
    WHERE "time"={"'"+time+"'"};"""
    cur.execute(load_model_query)
    r1=cur.fetchall()
    for row in r1:
        m=base64.b64decode(bytes(row[0]))
        m=bytes(m)
    cur.execute(load_preproc_query)
    r2=cur.fetchall()
    for row in r2:
        p=base64.b64decode(bytes(row[0]))
        p=bytes(p)
    model,preproc=pickle.loads(pickle.loads(m)),pickle.loads(pickle.loads(p))
    return preproc,model


print(get_title_basics_length())

random_blob=fetch_random_movie_info(10000)

preproc,model=train_cluster(random_blob)

time=save_model(pickle.dumps(preproc),pickle.dumps(model))

preproc1,model1=load_model(time)

print(model1.cluster_centers_)

conn.commit()
cur.close()

