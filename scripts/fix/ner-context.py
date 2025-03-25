import pandas as pd
import os

# MEDICAL ENTITIES dont need the ones in entities because they alr trained
batch_path = "./data/helsearkiv/batch/ner/final/"
csv_files = [os.path.join(batch_path, f) for f in os.listdir(batch_path) if f.endswith(".csv") and "b4" in f]

window_size = 50  # Tokens within the range of +/- 0

for file in csv_files:
    df = pd.read_csv(file)  # Read the CSV file into a dataframe
    for index, row in df.iterrows():
        current_text = str(row['Text'])
        tokens = current_text.split()  # Split the current row's text into tokens
        
        # Initialize the context list
        context_tokens = tokens[:]
        
        # Calculate the remaining token count needed to reach the window size
        remaining_tokens = window_size
        
        # Get previous tokens
        prev_index = index - 1
        while prev_index >= 0 and remaining_tokens > 0:
            prev_tokens = str(df.at[prev_index, 'Text']).split()
            if len(prev_tokens) <= remaining_tokens:
                context_tokens = prev_tokens + context_tokens
                remaining_tokens -= len(prev_tokens)
            else:
                context_tokens = prev_tokens[:remaining_tokens] + context_tokens
                break
            prev_index -= 1
        
        remaining_tokens = window_size

        # Get next tokens
        next_index = index + 1
        while next_index < len(df) and remaining_tokens > 0:
            next_tokens = str(df.at[next_index, 'Text']).split()
            if len(next_tokens) <= remaining_tokens:
                context_tokens += next_tokens
                remaining_tokens -= len(next_tokens)
            else:
                context_tokens += next_tokens[:remaining_tokens]
                break
            next_index += 1
        
        # Join the context tokens into a single string and assign to 'Context'
        df.at[index, 'Context'] = ' '.join(context_tokens)

    # Save the dataframe with the new 'Context' column (optional)
    df.to_csv(file, index=False)