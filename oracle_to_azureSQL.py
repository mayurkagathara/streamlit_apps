import pandas as pd
import pyodbc

# Read data from Oracle (replace with your Oracle connection details)
oracle_connection_string = "your_oracle_connection_string"
oracle_query = "SELECT * FROM your_oracle_table"

# Read data into a Pandas DataFrame
df = pd.read_sql(oracle_query, oracle_connection_string)

# Connect to Azure SQL (replace with your Azure SQL connection details)
azure_server = 'your_azure_server_name'
azure_database = 'your_azure_database_name'
azure_username = 'your_azure_username'
azure_password = 'your_azure_password'

azure_connection_string = f"Driver={{ODBC Driver 17 for SQL Server}};Server={azure_server}.database.windows.net;Database={azure_database};Uid={azure_username};Pwd={azure_password}"

# Insert DataFrame into Azure SQL
try:
    df.to_sql('YourAzureTable', con=azure_connection_string, if_exists='append', index=False)
    print("Data inserted successfully into Azure SQL!")
except Exception as e:
    print(f"Error: {str(e)}")
