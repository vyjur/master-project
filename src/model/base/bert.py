import os
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import cuda
from transformers import pipeline
from preprocess.setup import Preprocess
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
from model.util import Util
import wandb
from structure.enum import Task


class BERT:
    def __init__(
        self,
        task: Task,
        load: bool,
        save: str,
        dataset: list = [],
        tags_name: list = [],
        parameters: dict = {},
        tokenizer=None,
        project_name: str | None = None,
        pretrain: str | None = None,
    ):
        self.__device = "cuda:0" if cuda.is_available() else "cpu"
        
        if self.__device != "cpu":
            torch.cuda.set_device(self.__device)      
        print("Using:", self.__device, "with BERT")

        self.tokenizer = tokenizer
        self.__task = task

        if load:
            if task == Task.TOKEN:
                self.__model = AutoModelForTokenClassification.from_pretrained(
                    save, trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    save, trust_remote_code=True
                )
            else:
                self.__model = AutoModelForSequenceClassification.from_pretrained(
                    save, trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    save, trust_remote_code=True
                )

            print("Model and tokenizer loaded successfully.")
        else:
            wandb.init(project=f"{project_name}-{task}-bert-model".replace('"', ""))  # type: ignore
            wandb.config = {
                "learning_rate": parameters["learning_rate"],
                "epochs": parameters["epochs"],
                "batch_size": parameters["train_batch_size"],
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "logging_strategy": "epoch",
            }

            processed = Preprocess(
                self.tokenizer, parameters["max_length"]
            ).run_train_test_split(task, dataset, tags_name)
            class_weights = Util().class_weights(
                task, processed["dataset"], self.__device
            )

            torch.cuda.empty_cache()
            train_params = {
                "batch_size": parameters["train_batch_size"],
                "shuffle": parameters["shuffle"],
                "num_workers": parameters["num_workers"],
            }

            test_params = {
                "batch_size": parameters["valid_batch_size"],
                "shuffle": parameters["shuffle"],
                "num_workers": parameters["num_workers"],
            }

            training_loader = DataLoader(processed["train"], **train_params)
            testing_loader = DataLoader(processed["test"], **test_params)

            num_training_steps = len(training_loader)

            if len(processed['id2label']) != len(class_weights):
                # Original dictionary
                # Remove the key-value pair with key 0
                processed['id2label'].pop(0)

                # Update the keys to reduce them by 1
                processed['id2label'] = {key - 1: value for key, value in processed['id2label'].items()}
                
                del processed['label2id']['O']
                for tag in processed["label2id"]:
                    processed['label2id'][tag] -= 1  
                    

            if task == Task.TOKEN:
                self.__model = AutoModelForTokenClassification.from_pretrained(
                    pretrain,
                    trust_remote_code=True,
                    num_labels=len(processed["id2label"]),
                    id2label=processed["id2label"],
                    label2id=processed["label2id"],
                )
            else:
                self.__model = AutoModelForSequenceClassification.from_pretrained(
                    "ltg/norbert3-xs",
                    trust_remote_code=True,
                    num_labels=len(processed["id2label"]),
                    id2label=processed["id2label"],
                    label2id=processed["label2id"],
                )

            self.__model.to(self.__device)

            optimizer = torch.optim.Adam(  # type: ignore
                params=self.__model.parameters(), lr=parameters["learning_rate"]
            )

            lr_scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            for epoch in range(parameters["epochs"]):
                print(f"Training Epoch: {epoch}")
                loss, acc = self.__train(
                    training_loader,
                    num_training_steps,
                    optimizer,
                    lr_scheduler,
                    loss_fn,
                )
                wandb.log({"loss": loss, "accuracy": acc})  # type: ignore

            labels, predictions = self.__valid(
                testing_loader, self.__device, processed["id2label"]
            )
            Util().validate_report(labels, predictions)

            # Save the model
            if not os.path.exists(save):
                os.makedirs(save)
            self.__model.save_pretrained(save)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save)  # type:ignore
            wandb.finish()

        if task == Task.SEQUENCE:
            task_text = "text"
        else:
            task_text = "token"
        self.__pipeline = pipeline(
            task=f"{task_text}-classification",
            model=self.__model.to(self.__device),
            device=0 if self.__device == "gpu" else None,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

    def predict(self, data, pipeline=False):
        if pipeline:
            return self.__pipeline(data)
        else:
            self.__model = self.__model.to(self.__device)

            mask = np.where(np.array(data) == 0, 0, 1)
            outputs = self.__model(
                input_ids=torch.tensor(data, dtype=torch.long).to(self.__device),
                attention_mask=torch.tensor(mask, dtype=torch.long).to(self.__device),
            )
            if self.__task == Task.TOKEN:
                pred = torch.argmax(outputs.logits, dim=2)
            else:
                pred = torch.argmax(outputs.logits, dim=1).tolist()
                prob = [max(all_prob) for all_prob in nn.functional.softmax(outputs.logits, dim=-1)]
                return pred, prob
        return pred

    def __train(
        self,
        training_loader,
        num_training_steps,
        optimizer,
        scheduler=None,
        loss_fn=None,
    ):
        progress_bar = tqdm(range(num_training_steps))
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        self.__model.train()

        for _, batch in enumerate(training_loader):
            ids = batch["ids"].to(self.__device, dtype=torch.long)
            mask = batch["mask"].to(self.__device, dtype=torch.long)
            targets = batch["targets"].to(self.__device, dtype=torch.long)
            outputs = self.__model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            # tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # compute training accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(
                -1, self.__model.num_labels
            )  # shape (batch_size * seq_len, num_labels)

            loss = loss_fn(active_logits, flattened_targets)  # type: ignore

            tr_loss += loss
            # Now compute predictions based on the probabilities
            flattened_predictions = torch.argmax(
                active_logits,
                axis=1,  # type: ignore
            )  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            
            if self.__task == Task.TOKEN:
                active_accuracy = (
                    mask.view(-1) == 1
                )  # active accuracy is also of shape (batch_size * seq_len,)
                targets = flattened_targets[active_accuracy]  # Keep gradients
                predictions = flattened_predictions[active_accuracy]  # Keep gradients
            else:
                targets = flattened_targets
                predictions = flattened_predictions

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            tmp_tr_accuracy = accuracy_score(
                targets.numpy(), predictions.numpy()
            )
            tr_accuracy += tmp_tr_accuracy

            torch.nn.utils.clip_grad_norm_(
                parameters=self.__model.parameters(), max_norm=10
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            progress_bar.update(1)

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")
        return epoch_loss, tr_accuracy

    def __valid(self, testing_loader, device, id2label):
        # put model in evaluation mode
        self.__model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(testing_loader):
                ids = batch["ids"].to(device, dtype=torch.long)
                mask = batch["mask"].to(device, dtype=torch.long)
                targets = batch["targets"].to(device, dtype=torch.long)

                outputs = self.__model(
                    input_ids=ids, attention_mask=mask, labels=targets
                )
                loss, eval_logits = outputs.loss, outputs.logits

                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)

                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")

                # compute evaluation accuracy
                flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(
                    -1, self.__model.num_labels
                )  # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(
                    active_logits,
                    axis=1,  # type: ignore
                )  # shape (batch_size * seq_len,)
                # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                active_accuracy = (
                    mask.view(-1) == 1
                )  # active accuracy is also of shape (batch_size * seq_len,)
                if self.__task == Task.TOKEN:
                    targets = torch.masked_select(flattened_targets, active_accuracy)
                    predictions = torch.masked_select(
                        flattened_predictions, active_accuracy
                    )
                else:
                    targets = flattened_targets
                    predictions = flattened_predictions

                eval_labels.extend(targets)
                eval_preds.extend(predictions)

                tmp_eval_accuracy = accuracy_score(
                    targets.cpu().numpy(), predictions.cpu().numpy()
                )
                eval_accuracy += tmp_eval_accuracy

        labels = [id2label[id.item()] for id in eval_labels]
        predictions = [id2label[id.item()] for id in eval_preds]

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")

        return labels, predictions
