from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, AutoTokenizer
from torch import cuda
from transformers import pipeline
from preprocess.setup import Preprocess
from transformers import get_scheduler
from tqdm.auto import tqdm

SAVE_DIRECTORY = './src/model/saved_model'

class FineTunedBert:
    
    def __init__(self, load:bool=True, dataset:list=[], tags_name:list=[]):
        device = 'cuda' if cuda.is_available() else 'cpu'
        print("Using:", device)
        
        if load:
            self.__model = AutoModelForTokenClassification.from_pretrained(SAVE_DIRECTORY)
            self.__tokenizer = AutoTokenizer.from_pretrained(SAVE_DIRECTORY)
            
            print("Model and tokenizer loaded successfully.")
        else:

            checkpoint = 'distilbert-base-cased'
            self.__tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            processed = Preprocess(self.__tokenizer).run(dataset, tags_name)

            # TODO: move this out and fix device
            TRAIN_BATCH_SIZE = 2
            VALID_BATCH_SIZE = 2
            EPOCHS = 3
            LEARNING_RATE = 1e-05
            MAX_GRAD_NORM = 10

            train_params = {'batch_size': TRAIN_BATCH_SIZE,
                            'shuffle': True,
                            'num_workers': 0
                            }

            test_params = {'batch_size': VALID_BATCH_SIZE,
                            'shuffle': True,
                            'num_workers': 0
                            }
            
            training_loader = DataLoader(processed['train'], **train_params)
            testing_loader = DataLoader(processed['test'], **test_params)

            num_training_steps = len(training_loader)

            self.__model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(processed['id2label']), id2label=processed['id2label'], label2id = processed['label2id'])
            self.__model.to(device)

            optimizer = torch.optim.Adam(params=self.__model.parameters(), lr=LEARNING_RATE)

            # TODO
            lr_scheduler = get_scheduler(
                name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )

            for epoch in range(EPOCHS):
                print(f"Training Epoch: {epoch}")
                self.__train(training_loader, num_training_steps, device, optimizer)

            labels, predictions = self.__valid(testing_loader, device, processed['id2label'])
            print(classification_report(labels, predictions))

            # Save the model
            self.__model.save_pretrained(SAVE_DIRECTORY)

            # Save the tokenizer
            self.__tokenizer.save_pretrained(SAVE_DIRECTORY)
                        
        self.__pipeline = pipeline(task="token-classification", model=self.__model.to(device), device=0, tokenizer=self.__tokenizer, aggregation_strategy="simple")
            
    def predict(self, data):
        return self.__pipeline(data)
    
    def __train(self, training_loader, num_training_steps, device, optimizer):
        progress_bar = tqdm(range(num_training_steps))
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        self.__model.train()
        
        for idx, batch in enumerate(training_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            outputs = self.__model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if idx % 100==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")
            
            # compute training accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.__model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            tr_preds.extend(predictions)
            tr_labels.extend(targets)
            
            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    def __valid(self, testing_loader, device, id2label):
        # put model in evaluation mode
        self.__model.eval()
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []
        
        with torch.no_grad():
            for idx, batch in enumerate(testing_loader):
                
                ids = batch['ids'].to(device, dtype = torch.long)
                mask = batch['mask'].to(device, dtype = torch.long)
                targets = batch['targets'].to(device, dtype = torch.long)
                
                outputs = self.__model(input_ids=ids, attention_mask=mask, labels=targets)
                loss, eval_logits = outputs.loss, outputs.logits
                
                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)
            
                if idx % 100==0:
                    loss_step = eval_loss/nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")
                
                # compute evaluation accuracy
                flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(-1, self.__model.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
                targets = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                eval_labels.extend(targets)
                eval_preds.extend(predictions)
                
                tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy
        

        labels = [id2label[id.item()] for id in eval_labels]
        predictions = [id2label[id.item()] for id in eval_preds]

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")

        return labels, predictions
    
    
if __name__ == '__main__':
    import json

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

    model = FineTunedBert(False, dataset_sample, tags)


    
