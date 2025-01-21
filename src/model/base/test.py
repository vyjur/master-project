import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertLSTMModel(nn.Module):
    def __init__(
        self,
        bert_model,
        lstm_hidden_dim,
        num_classes,
        lstm_num_layers=1,
        bidirectional=True,
    ):
        super(BertLSTMModel, self).__init__()
        self.bert = bert_model  # Pre-trained BERT model
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=bert_model.config.hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state (sequence_output) from BERT
        bert_hidden_states = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, seq_len, hidden_size)

        # Pass through LSTM
        lstm_output, (h_n, c_n) = self.lstm(bert_hidden_states)  # LSTM outputs
        # Optionally, use the final LSTM hidden state for classification
        if self.bidirectional:
            # Concatenate the final states from both directions
            final_hidden_state = torch.cat(
                (h_n[-2], h_n[-1]), dim=1
            )  # Shape: (batch_size, lstm_hidden_dim*2)
        else:
            final_hidden_state = h_n[-1]  # Shape: (batch_size, lstm_hidden_dim)

        # Pass the LSTM final hidden state through the fully connected layer
        output = self.fc(final_hidden_state)  # Shape: (batch_size, num_classes)

        return output


# Example Usage
if __name__ == "__main__":
    # Load tokenizer and BERT model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # Define the model with LSTM
    lstm_hidden_dim = 128
    num_classes = 2  # Example: Binary classification
    model = BertLSTMModel(bert_model, lstm_hidden_dim, num_classes)

    # Tokenize example input
    text = ["Example sentence for BERT embedding followed by LSTM."]
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    print(output)
    print("HALLO")
