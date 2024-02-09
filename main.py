from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import text
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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

"""
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, ["B"])
    ]
)
"""

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
    return df


def fetch_random_movie_info(length): #migth be shorter than the specified length
    x=1000
    def query(x):
        query=f"""
        CREATE EXTENSION IF NOT EXISTS tsm_system_rows;
        CREATE OR REPLACE VIEW title_random AS 
            SELECT *
            FROM title_basics
            TABLESAMPLE SYSTEM_ROWS({x});
        SELECT DISTINCT ON ("tconst") "isAdult", "startYear", "nconst", "genres", "runtimeMinutes", "language", "directors", "writers", "characters", "tconst"
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
    if length%1000 != 0 : 
        x=length%1000
        df=pd.concat([df,pd.read_sql(query(x),db)])
    return df.drop_duplicates()
    
#blob=fetch_movie_info(length/100,1000)
random_blob=fetch_random_movie_info(10001)

"""
print(blob.head())
print(blob.shape[0])"""
print(random_blob.shape[0])