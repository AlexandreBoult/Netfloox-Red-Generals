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
    echo=True,
).connect()
n_rows=100
cluster="none"
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

categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, ["B"])
    ]
)

title_blob=pd.read_sql(join_query,db).drop(columns=["ordering"])
print(title_blob)