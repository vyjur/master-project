# NOTE: Similar to BiLSTM file but includes CRF. Did not base this on BiLSTM because it
# Based on https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
from model.base.nn import NN
from structure.enum import Task
from model.nn.bilstm import Model as BaseModel

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Model(BaseModel):
    def __init__(self, batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim, bert_model=None):
        super(Model, self).__init__(
            batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim, bert_model
        )

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
            torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
        )

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.0).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).to(self.device)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].to(self.device)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat(
            [
                torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(
                    self.device
                ),
                tags,
            ]
        )
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.0).to(self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag].to(self.device)
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  # type: ignore
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].to(self.device)
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][int(best_tag_id)]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = [self._forward_alg(feat) for feat in feats]
        gold_score = [self._score_sentence(feat, tag) for feat, tag in zip(feats, tags)]
        return sum(forward_score) / len(forward_score) - sum(gold_score) / len(
            gold_score
        )

    def _get_lstm_features(self, sentences):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentences)
        if self.bert:
            embeds = embeds.last_hidden_state
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emissiwn scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        
        sum_score = self._forward_alg(lstm_feats)

        # Find the best path, given the features.
        result = [self._viterbi_decode(lstm_feat) for lstm_feat in lstm_feats]
        score = torch.tensor([res[0] for res in result]).to(self.device)
        tag_seq = [res[1] for res in result]
        
        prob = (score - sum_score)/len(score)
        
        print("TAG", tag_seq)
        print("SCORE", score)
        print("prob", prob)
        return torch.tensor(tag_seq).to(self.device), prob


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        load: bool,
        save: str,
        dataset: list = [],
        tags_name: list = [],
        parameters: dict = {},
        tokenizer=None,
        project_name: str | None = None,
        pretrain: str | None = None,
    ):
        super(BiLSTMCRF, self).__init__()

        self.__model = NN(
            Model,  # type: ignore
            Task.TOKEN,
            load,
            save,
            dataset,
            tags_name,
            parameters,
            tokenizer,
            project_name,
            pretrain,
        )
        self.tokenizer = self.__model.tokenizer
        self.device = self.__model.device

    def predict(self, data, pipeline=False):
        return self.__model.predict(data, pipeline)

    def forward(self, x):
        return self.__model(x)
    
if __name__ == "__main__":
    import os
    from preprocess.dataset import DatasetManager
    from structure.enum import Dataset

    train_parameters = {
        "train_batch_size": 2,
        "valid_batch_size": 2,
        "epochs": 3,
        "learning_rate": 1e-04,
        "shuffle": True,
        "num_workers": 0,
        "max_length": 128,
    }

    checkpoint = "ltg/norbert3-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    folder_path = "./data/annotated/"
    files = [
        folder_path + f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    manager = DatasetManager(files)
    raw_dataset = manager.get(Dataset.NER)

    save = "./test_model"

    dataset = []
    tags = set()
    for doc in raw_dataset:
        curr_doc = []
        for row in doc.itertuples(index=False):
            curr_doc.append((row[2], row[3]))  # Add (row[1], row[2]) tuple to list
            tags.add(row[3])  # Add row[2] to the set

        dataset.append(curr_doc)
    tags = list(tags)

    model = BiLSTMCRF(
        False, save, dataset, tags, train_parameters, tokenizer, "test_wandb_"
    )
    tokenized = Preprocess(model.tokenizer).run(["Hei på deg!"])  # type: ignore

    pred1 = model.predict([tokenized[0]["input_ids"], tokenized[1]["input_ids"]])
    pred2 = model.predict([tokenized[0]["input_ids"]])

    print(len(tokenized[0]["input_ids"]), len(pred1[0]))
    print(len(tokenized[1]["input_ids"]), len(pred1[1]))
    print(len(tokenized[0]["input_ids"]), len(pred2[0]))
