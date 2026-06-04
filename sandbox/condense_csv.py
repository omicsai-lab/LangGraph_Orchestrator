import pandas as pd
import random

input_file = "bc_counts_transposed.csv"
output_file = "bc_counts_transposed_condensed.csv"

# 1. Use pandas to read JUST the first row. 
# This automatically handles empty strings and names them 'Unnamed: 0' safely.
all_columns = pd.read_csv(input_file, nrows=0).columns.tolist()

# The 0th column is your index column (will be 'Unnamed: 0')
index_column_name = all_columns[0]
data_columns = all_columns[1:]

# 2. Specify your target genomics/count data columns
must_have_columns = ["BTN1A1", "OLAH", "LILRB5", "CSF1R", "CEL","CELP", "LALBA", "CSN2","LINC02532","LPO"] 

# 3. Separate and randomly sample the remaining data columns
remaining_columns = [col for col in data_columns if col not in must_have_columns]
sample_size = min(100, len(remaining_columns))
random_sampled_columns = random.sample(remaining_columns, sample_size)

# 4. Construct the final exact column layout (Index + Targets + Randoms)
final_columns_to_keep = [index_column_name] + must_have_columns + random_sampled_columns

# 5. Read the massive file EXACTLY ONCE using the column name array
print("Loading data... (This will be much faster now)")
df_condensed = pd.read_csv(input_file, usecols=final_columns_to_keep)

# 6. Reorder data and save it with the index column locked at the front
df_condensed = df_condensed[final_columns_to_keep]
df_condensed.to_csv(output_file, index=False)

print(f"Done! Saved index + {len(must_have_columns) + sample_size} columns to {output_file}")
