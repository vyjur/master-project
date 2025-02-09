import torch
import torch.nn as nn

def compute_mnlp(sentence, model):
    sentence = [val.ids for val in sentence]
    if hasattr(model, 'device'):
        tokens = torch.tensor(sentence, dtype=torch.long).to(model.device)  # Exclude the last token
    else:
        tokens = sentence
    
    if hasattr(model, 'batch'):
        model.batch = tokens.shape[0]
    
    with torch.no_grad():
        # INFO: The normalized log probability is computed in the model itself
        output, prob = model(tokens)
        print(output.shape, prob.shape)
    return sum(prob)/len(output)