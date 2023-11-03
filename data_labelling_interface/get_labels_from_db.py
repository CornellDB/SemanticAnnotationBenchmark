import pandas as pd
from sqlalchemy import create_engine

# Define the SQLite database connection string
db_path = "semantic_annotation_backend.db"
engine = create_engine(f"sqlite:///{db_path}")

# Write an SQL query to retrieve the data
sql_query = "SELECT * FROM labels"

# Execute the query and load the results into a Pandas DataFrame
df = pd.read_sql(sql_query, con=engine)

# Now, you can work with the data in the DataFrame (df)
df.to_csv("labels.csv")
