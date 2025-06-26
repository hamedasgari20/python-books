# 📁 FILE: data_preparation.py
# PURPOSE: This script collects Python-related questions and answers from Stack Overflow using Google BigQuery,
#          cleans the text data, splits it into training and evaluation sets, and saves them as JSON files.
# TOOL: Google BigQuery (GCP), pandas, scikit-learn
# USE CASE: Preparing Q&A dataset for fine-tuning LLMs in an LLMOps pipeline

import os
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

# Set GCP credentials
# 🔐 Step 1: Set up Google Cloud authentication
# --------------------------------------------------------------------------------------------------
# You must have a valid GCP service account JSON key file to access BigQuery.
# Make sure the JSON file has proper permissions on the Stack Overflow public dataset.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/HP i5 Series/Desktop/service-account.json"
client = bigquery.Client(project="fastapi-login-420407")

# 📊 Step 2: Define SQL query to fetch Stack Overflow Python Q&A pairs
# --------------------------------------------------------------------------------------------------
# This query retrieves:
# - Questions (`title` + `body`) as input text
# - Accepted answers (`body`) as output text
# It filters:
# - Only questions with accepted answers
# - Questions tagged with 'python'
# - Answers created after Jan 1, 2020
# LIMIT is set to 10 for demo; increase for real training.

QUERY = """
SELECT
    CONCAT(q.title, q.body) AS input_text,
    a.body AS output_text
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
ON
    q.accepted_answer_id = a.id
WHERE
    q.accepted_answer_id IS NOT NULL AND
    REGEXP_CONTAINS(q.tags, r"python") AND
    a.creation_date >= "2020-01-01"
LIMIT
    10
"""

# 🔍 Step 3: Execute query and convert result to Pandas DataFrame
print("🔍 Fetching data from BigQuery...")
query_job = client.query(QUERY)
df = query_job.to_dataframe()

# 🧹 Step 4: Clean HTML tags and extra spaces from text fields
# --------------------------------------------------------------------------------------------------
# Stack Overflow posts often include HTML formatting (e.g., <code>, <p>). We remove these to clean inputs.
# We also strip leading/trailing whitespace to ensure consistency.

df['input_text'] = df['input_text'].str.replace(r'<[^>]+>', '', regex=True).str.strip()
df['output_text'] = df['output_text'].str.replace(r'<[^>]+>', '', regex=True).str.strip()

# 📦 Step 5: Split dataset into training and evaluation subsets
# --------------------------------------------------------------------------------------------------
# We use an 80/20 split for training and evaluation.
# Using a fixed random_state ensures reproducibility across runs.

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# 💾 Step 6: Save datasets to disk in JSON format
# --------------------------------------------------------------------------------------------------
# We save both datasets as JSON files for easy loading during model training.
# Each record is stored as a JSON object, preserving the structure.

train_df.to_json("train_data.json", orient="records")
eval_df.to_json("eval_data.json", orient="records")

# ✅ Step 7: Confirm successful completion
print("✅ Data saved to train_data.json and eval_data.json")
