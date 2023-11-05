from datetime import datetime
import os
from dotenv import dotenv_values
import pandas as pd
import boto3

from sqlalchemy import create_engine

dirname = os.path.dirname(__file__)
config = dotenv_values(f"{dirname}/../.env")
s3 = boto3.client(
    "s3", aws_access_key_id=config["AWS_ACCESS_KEY_ID"], aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)

# Define the SQLite database connection string
db_path = f"{dirname}/semantic_annotation_backend.db"
engine = create_engine(f"sqlite:///{db_path}")

# Write an SQL query to retrieve the data
sql_query = "SELECT * FROM labels"

# Execute the query and load the results into a Pandas DataFrame
df = pd.read_sql(sql_query, con=engine)

# Now, you can work with the data in the DataFrame (df)
csv_name = f"labels_{datetime.utcnow()}.csv"
df.to_csv(f"{dirname}/labels/{csv_name}")
s3.upload_file(f"{dirname}/labels/{csv_name}", "semantic-annotation-tables", f"labels/{csv_name}")
