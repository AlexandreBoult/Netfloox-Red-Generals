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
n_rows=1000
cluster="none"
offset=0
join_query=f"""
    WITH title_basics_l AS (SELECT * from title_basics LIMIT {n_rows})
    SELECT DISTINCT ON ("tconst") *
    FROM title_basics_l
        LEFT JOIN title_akas
            ON "tconst" = title_akas."titleId"
        LEFT JOIN title_crew
            USING("tconst")
        LEFT JOIN title_ratings
            USING("tconst")
        LEFT JOIN title_principals
            USING("tconst")
    --WHERE "cluster" == {cluster};
"""

def title_query(n_rows,offset):
    cluster_feed_query=f"""
        WITH title_basics_l AS (SELECT * from title_basics LIMIT {n_rows} OFFSET {offset})
        SELECT DISTINCT ON ("tconst") "isAdult", "startYear", "nconst", "genres", "runtimeMinutes", "language", "directors", "writers", "characters", "tconst"
        FROM title_basics_l
            LEFT JOIN title_akas
                ON "tconst" = title_akas."titleId"
            LEFT JOIN title_crew
                USING("tconst")
            LEFT JOIN title_ratings
                USING("tconst")
            LEFT JOIN title_principals
                USING("tconst");
    """
    return cluster_feed_query

categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, ["B"])
    ]
)

get_length="""
SELECT COUNT(DISTINCT "tconst") FROM title_basics;
"""
length=pd.read_sql(get_length,db)["count"][0]
print(length)

df=pd.DataFrame()
for n in range(length//1000):
    df=pd.concat([df,pd.read_sql(title_query(1000,n*1000),db)])
    print(f"{n/(length//1000)*100} % done")
print(df.head())