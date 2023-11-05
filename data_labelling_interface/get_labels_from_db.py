from datetime import datetime
from dotenv import dotenv_values
import pandas as pd
import boto3

from sqlalchemy import create_engine

config = dotenv_values("../.env")
s3 = boto3.client(
    "s3", aws_access_key_id=config["AWS_ACCESS_KEY_ID"], aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)

# Define the SQLite database connection string
db_path = "semantic_annotation_backend.db"
engine = create_engine(f"sqlite:///{db_path}")

# Write an SQL query to retrieve the data
sql_query = "SELECT * FROM labels"

# Execute the query and load the results into a Pandas DataFrame
df = pd.read_sql(sql_query, con=engine)

# Now, you can work with the data in the DataFrame (df)
csv_name = f"labels_{datetime.utcnow()}.csv"
df.to_csv(csv_name)
s3.upload_file(csv_name, "semantic-annotation-tables", f"labels/{csv_name}")
