import streamlit as st
import pandas as pd
import csv
#from io import StringIO

# Define the path to the CSV file
PATH_1 = "path/to/your/csv/file.csv"

@st.cache  # Cache the data loading
def load_data():
    return pd.read_csv(PATH_1)

@st.cache  # Cache the data loading
def load_data():
    with open(PATH_1, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Assuming the first row is the header
        data = [row + [row[1].split(',', 1)[1]] if len(row) > 1 else row for row in csvreader]

    return pd.DataFrame(data, columns=header)

# Load data
data = load_data()

# Sidebar for user input
st.sidebar.header("Search Data")
search_term = st.sidebar.text_input("Enter search term:")

# Filter data based on search term
if search_term:
    filtered_data = data[data.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)]
else:
    filtered_data = data

# Display the filtered data
st.write("### Displaying Filtered Data")
st.write(filtered_data)
