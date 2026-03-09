import pandas as pd

df = pd.read_csv('data/01_full_sample_with_prompts.csv', low_memory=False)

# Look for question related columns. 
# Based on grep results, columns like 'estas', 'lernas', 'flugas' contain the translated answers.
# Let's see if we can find the prompt texts.
# In many Qualtrics exports, the second or third row contains the question text.

# Print the first few rows of some columns to identify them
potential_cols = [col for col in df.columns if 'estas' in str(df[col].iloc[0]) or 'lernas' in str(df[col].iloc[0])]
print("Potential columns:", potential_cols)

# Just print the first 100 columns to see what we have
print("First 100 columns:", df.columns[:100].tolist())

# It seems the data is already flattened.
# Let's try to extract unique values from columns that look like they might be part of the task.
# Looking at the grep output:
# "tom ne dormas,anna ne lernas,la hundo ne flugas,le birdo ne estas ganda"
# This looks like a set of answers.

# I'll search for the original English prompts if they exist.
# "Tom does not sleep", "Anna does not learn", etc.
