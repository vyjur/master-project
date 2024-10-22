import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
from sklearn.metrics import accuracy_score, classification_report
from util.setup import Util

from pipeline.lexicon import Lexicon

SAVE_DIRECTORY = './src/ner/saved/bilstmcrf'

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
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class Model(nn.Module):

    def __init__(self, batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.batch = batch
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        #TODO
        return (torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
            torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device))
        
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentences):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentences)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score


    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
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
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

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
        return sum(forward_score)/len(forward_score) - sum(gold_score)/len(gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        # score, tag_seq = self._viterbi_decode(lstm_feats)
        result = [self._viterbi_decode(lstm_feat)[1] for lstm_feat in lstm_feats]
        return torch.tensor(result).to(self.device)


class BiLSTMCRF:

    def __init__(self, load: bool = True, dataset: list = [], tags_name: list = [], parameters: dict = [], align:bool = True, tokenizer = None):
        self.__device = "cuda" if cuda.is_available() else "cpu"
        print("Using:", self.__device)

        self.__valid_batch_size = parameters['valid_batch_size']

        self.tokenizer = tokenizer

        vocab_size = self.tokenizer.vocab_size
        # TODO
        # embedding_dim = self.tokenizer.model_max_length
        embedding_dim = 300
        processed = Preprocess(self.tokenizer).run_train_test_split(dataset, tags_name, align)
        class_weights = Preprocess(self.tokenizer).class_weights(processed['dataset'], self.__device)

        tag_to_ix = processed['label2id']
        START_ID = max(processed['id2label'].keys()) + 1
        STOP_ID = max(processed['id2label'].keys())+2

        tag_to_ix[START_TAG] = START_ID
        tag_to_ix[STOP_TAG] = STOP_ID

        # TODO
        hidden_dim = parameters['max_length']
        n_tags = len(tags_name)

        if load:
            self.__model = Model(1, vocab_size, tag_to_ix, embedding_dim, hidden_dim)
            self.__model.load_state_dict(torch.load(SAVE_DIRECTORY+'/model.pth', weights_only=True))
        else:

            train_params = {'batch_size': parameters['train_batch_size'],
                            'shuffle': True,
                            'num_workers': 0
                            }

            test_params = {'batch_size': parameters['valid_batch_size'],
                            'shuffle': True,
                            'num_workers': 0
                            }

            training_loader = DataLoader(processed["train"], **train_params)
            testing_loader = DataLoader(processed["test"], **test_params)

            self.__model = Model(parameters['train_batch_size'], vocab_size, tag_to_ix, embedding_dim, hidden_dim).to(self.__device)

            loss_fn = nn.CrossEntropyLoss(weight = class_weights)
            # loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.__model.parameters(), lr=parameters['learning_rate'], weight_decay=1e-4)

            for t in range(parameters['epochs']):
                print(f"Epoch {t+1}\n-------------------------------")
                self.__train(training_loader, loss_fn, optimizer)
                # self.__test(testing_loader, loss_fn)

            labels, predictions = self.__valid(testing_loader, self.__device, processed['id2label'])
            lexi_predictions = Lexicon().predict(processed['test_raw'], self.tokenizer)
            lexi_predictions = Lexicon().merge(lexi_predictions, predictions)
            Util().validate_output(labels, predictions, lexi_predictions)

            torch.save(self.__model.state_dict(), SAVE_DIRECTORY + "/model.pth")
        
    def predict(self, data, pipeline=False):
        data_tensor = torch.tensor(data, dtype=torch.long).to(self.__device)
        self.__model.batch = data_tensor.shape[0]
        return self.__model(data_tensor)

    def __train(self, training_loader, loss_fn, optimizer):
        self.__model.train()

        for idx, batch in enumerate(training_loader):
            ids = batch['ids'].to(self.__device, dtype=torch.long)
            # mask = batch['mask'].to(self.__device, dtype=torch.bool)
            targets = batch['targets'].to(self.__device, dtype=torch.long)
            self.__model.batch = ids.shape[0]

            self.__model.zero_grad()

            # Step 3. Run our forward pass.
            loss = self.__model.neg_log_likelihood(ids, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:  # Adjust as needed
                print(f"Batch {idx}, Loss: {loss.item()}")   
                pass

    def __valid(self, testing_loader, device, id2label):
        # put model in evaluation mode
        self.__model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(testing_loader):
                ids = batch['ids'].to(device, dtype = torch.long)
                # mask = batch['mask'].to(device, dtype = torch.long)
                targets = batch['targets'].to(device, dtype = torch.long)
                self.__model.batch = ids.shape[0]
                outputs = self.__model(ids)

                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)

                # compute evaluation accuracy
                flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
                flattened_predictions = outputs.view(-1) # shape (batch_size * seq_len,)

                # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                eval_labels.extend(flattened_targets.tolist())
                eval_preds.extend(flattened_predictions.tolist())

                tmp_eval_accuracy = accuracy_score(flattened_targets.cpu().numpy(), flattened_predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy

        labels = [id2label[id] for id in eval_labels]
        predictions = [id2label[id] for id in eval_preds]

        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Accuracy: {eval_accuracy}")

        return labels, predictions

if __name__ == '__main__':
    import json
    
    train_parameters = {
        'train_batch_size': 2,
        'valid_batch_size': 2,
        'epochs': 1,
        'learning_rate': 1e-04,
        'shuffle': True,
        'num_workers': 0
    }

    with open('./data/Corona2.json') as f:
        d = json.load(f)

    dataset_sample = []

    for example in d['examples']:
        
        entities = [ (annot['start'], annot['end'], annot['value'], annot['tag_name']) for annot in example['annotations']]
        
        dataset_sample.append({
            'text': example['content'],
            'entities': entities
        })

    tags = set()

    for example in d['examples']:
        for annot in example['annotations']:
            tags.add(annot['tag_name'])
            
    tags = list(tags)
    print(tags)
    
    checkpoint = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = BiLSTMCRF(False, dataset_sample, tags, train_parameters, tokenizer)
    tokenized = Preprocess(model.tokenizer).run([dataset_sample[0], dataset_sample[1]])
    
    pred1 = model.predict([tokenized[0]['input_ids'], tokenized[1]['input_ids']])
    pred2 = model.predict([tokenized[0]['input_ids']])
    
    print(len(tokenized[0]['input_ids']), len(pred1[0]))
    print(len(tokenized[1]['input_ids']), len(pred1[1]))
    print(len(tokenized[0]['input_ids']), len(pred2[0]))
