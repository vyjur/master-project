import os
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import cuda
from transformers import pipeline
from preprocess.setup import Preprocess
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
from model.regularization.early_stopping import EarlyStopping
from model.util import Util
import wandb
from structure.enum import Task
from model.tuning.setup import TuningConfig
import gc

import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

sweep_config = TuningConfig.get_config()


class BERT(nn.Module):
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
        util: Util = None
    ):
        super(BERT, self).__init__()

        self.__device = "cuda:0" if cuda.is_available() else "cpu"

        if self.__device != "cpu":
            torch.cuda.set_device(self.__device)

        print("Using:", self.__device, "with BERT")

        self.device = self.__device

        self.tokenizer = tokenizer
        self.__task = task
        self.__parameters = parameters
        self.__pretrain = pretrain
        self.__save = save
        self.__dataset = dataset
        self.__tags_name = tags_name
        self.__project_name = project_name
        self.__util = util if util is not None else Util()
        
        if load:
            if task == Task.TOKEN:
                self.__model = AutoModelForTokenClassification.from_pretrained(
                    save, trust_remote_code=True, device_map=self.__device
                ).to(self.__device)
            else:
                self.__model = AutoModelForSequenceClassification.from_pretrained(
                    save, trust_remote_code=True, device_map=self.__device
                ).to(self.__device)
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                save, trust_remote_code=True, device_map=self.__device
            )
            
            self.__processed = Preprocess(
                self.tokenizer, parameters["max_length"], parameters["stride"], self.__util
            ).run_train_test_split(self.__task, self.__dataset, self.__tags_name)

            # try:
            #     print("### Extra set performance:")
            #     test_dataset = torch.load("./data/helsearkiv/test_dataset/test_dataset.pth")
            #     train_params = {
            #         "batch_size": parameters["train_batch_size"],
            #         "shuffle": parameters["shuffle"],
            #         "num_workers": parameters["num_workers"],
            #     }
            #     extra_loader = DataLoader(test_dataset, **train_params)
            #     loss_fn = nn.CrossEntropyLoss() 
            #     labels, predictions = self.__valid(
            #         extra_loader, loss_fn, self.__processed["id2label"], True
            #     )
            #     self.__util.validate_report(labels, predictions)
            # except Exception as e:
            #     print(e)

            print("Model and tokenizer loaded successfully.")
        else:

            self.__processed = None
            self.__class_weights = None

            tune = parameters["tune"]
            if tune:
                sweep_config['parameters']['valid_batch_size'] = {
                    "value": parameters['valid_batch_size']
                } 
                sweep_config['parameters']['shuffle'] = {
                    "value": parameters["shuffle"]
                }
                sweep_config['parameters']['num_workers'] = {
                    "value": parameters["num_workers"]
                }
                sweep_id = wandb.sweep(
                    sweep_config,
                    project=f"{project_name}-{task}-bert-model".replace('"', ""),
                )
                
                wandb.agent(sweep_id, self.train, count=parameters["tune_count"])
                
            else:
                wandb.config = {
                    "learning_rate": parameters["learning_rate"],
                    "epochs": parameters["epochs"],
                    "batch_size": parameters["train_batch_size"],
                    "valid_batch_size": parameters["valid_batch_size"],
                    "learning_rate": parameters["learning_rate"],
                    "optimizer": parameters["optimizer"],
                    "weight_decay": parameters["weight_decay"],
                    "early_stopping_patience": parameters["early_stopping_patience"],
                    "early_stopping_delta": parameters["early_stopping_delta"],
                    'max_length': parameters['max_length'],
                    'stride': parameters['stride'],
                    'shuffle': parameters['shuffle'],
                    'num_workers': parameters['num_workers'],
                    "evaluation_strategy": "epoch",
                    "save_strategy": "epoch",
                    "logging_strategy": "epoch",
                    "weights": parameters['weights'],
                    "tune": tune,
                }
                self.train(wandb.config)

        if task == Task.SEQUENCE:
            task_text = "text"
        else:
            task_text = "token"

        self.__pipeline = pipeline(
            task=f"{task_text}-classification",
            model=self.__model,
            device=0 if self.__device == "gpu" else None,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )
        
        

    def train(self, config = None):
        
        with wandb.init(project=f"{self.__project_name}-{self.__task}-bert-model".replace('"', "")):  # type: ignore
            if config is None:
                config = wandb.config

            self.__processed = Preprocess(
                self.tokenizer, config["max_length"], config["stride"], self.__util
            ).run_train_test_split(self.__task, self.__dataset, self.__tags_name)

            self.__class_weights = self.__util.class_weights(
                self.__task, self.__processed["dataset"], self.__device
            )
            
            train_params = {
                "batch_size": config["batch_size"],
                "shuffle": self.__parameters["shuffle"],
                "num_workers": self.__parameters["num_workers"],
            }

            test_params = {
                "batch_size": self.__parameters["valid_batch_size"],
                "shuffle": self.__parameters["shuffle"],
                "num_workers": self.__parameters["num_workers"],
            } 

            training_loader = DataLoader(self.__processed["train"], **train_params)
            valid_loader = DataLoader(self.__processed["valid"], **test_params)
            testing_loader = DataLoader(self.__processed["test"], **test_params)

            num_training_steps = len(training_loader)
            
            if hasattr(self, '__model'):
                del self.__model
            gc.collect()
            torch.cuda.empty_cache()

            if self.__task == Task.TOKEN:
                self.__model = AutoModelForTokenClassification.from_pretrained(
                    self.__pretrain,
                    trust_remote_code=True,
                    num_labels=len(self.__processed["id2label"]),
                    id2label=self.__processed["id2label"],
                    label2id=self.__processed["label2id"],
                ).to(self.__device)
            else:
                self.__model = AutoModelForSequenceClassification.from_pretrained(
                    self.__pretrain,
                    trust_remote_code=True,
                    num_labels=len(self.__processed["id2label"]),
                    id2label=self.__processed["id2label"],
                    label2id=self.__processed["label2id"],
                ).to(self.__device)

            if config["optimizer"] == "adam":
                optimizer = torch.optim.Adam(  # type: ignore
                    params=self.__model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                )
            elif config["optimizer"] == "sgd":
                optimizer = torch.optim.SGD(  # type: ignore
                    params=self.__model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                )
            else:
                optimizer = None

            early_stopping = EarlyStopping(
                patience=config["early_stopping_patience"],
                delta=config["early_stopping_delta"],
            )

            lr_scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
            
            if config['weights']:
                loss_fn = nn.CrossEntropyLoss(weight=self.__class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()

            for epoch in range(config["epochs"]):
                print(f"Training Epoch: {epoch}")
                train_loss, train_acc = self.__train(
                    training_loader,
                    num_training_steps,
                    optimizer,
                    lr_scheduler,
                    loss_fn,
                )

                val_loss, val_acc, macro_f1, weighted_f1 = self.__valid(
                    valid_loader, loss_fn, self.__processed["id2label"]
                )

                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1
                    }
                )  # type: ignore

                early_stopping(val_loss, self.__model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("### Valid set performance:")
            labels, predictions = self.__valid(
                valid_loader, loss_fn, self.__processed["id2label"], True
            )
            self.__util.validate_report(labels, predictions)

            print("### Test set performance:")
            labels, predictions = self.__valid(
                testing_loader, loss_fn, self.__processed["id2label"], True
            )
            self.__util.validate_report(labels, predictions)
            
            # try:
            #     print("### Extra set performance:")
            #     test_dataset = torch.load("./data/helsearkiv/test_dataset/test_dataset.pth")
            #     extra_loader = DataLoader(test_dataset, **train_params)
                
            #     labels, predictions = self.__valid(
            #         extra_loader, loss_fn, self.__processed["id2label"], True
            #     )
            #     self.__util.validate_report(labels, predictions)
            # except Exception as e:
            #     print(e)
            
            if "intra" in self.__processed and "inter" in self.__processed:
                intra_loader = DataLoader(self.__processed["intra"], **train_params)
                inter_loader = DataLoader(self.__processed["inter"], **train_params)

                print("### Inter sentences performance:")
                labels, predictions = self.__valid(
                    inter_loader, loss_fn, self.__processed["id2label"], True
                )
                self.__util.validate_report(labels, predictions)
                
                print("### Intra sentences performance:")
                labels, predictions = self.__valid(
                    intra_loader, loss_fn, self.__processed["id2label"], True
                )
                self.__util.validate_report(labels, predictions)

            # If tune don't save else too many models heavy
            if not config['tune']:
                # Save the model
                if not os.path.exists(self.__save):
                    os.makedirs(self.__save)
                self.__model.save_pretrained(self.__save)

                # Save the tokenizer
                self.tokenizer.save_pretrained(self.__save)  # type:ignore
            else:
                pass
                # TODO: save
                if not os.path.exists(f"{self.__save}/{wandb.run.id}"):
                    os.makedirs(f"{self.__save}/{wandb.run.id}")            
                self.__model.save_pretrained(f"{self.__save}/{wandb.run.id}")
                self.tokenizer.save_pretrained(f"{self.__save}/{wandb.run.id}")  # type:ignore
                
                
    def predict(self, data, pipeline=False):
        self.__model.eval()
        
        if pipeline:
            return self.__pipeline(data)
        else:
            
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                mask = np.where(np.array(data) == 3, 0, 1)
                outputs = self.__model(
                    input_ids=torch.tensor(data, dtype=torch.long).to(self.__device),
                    attention_mask=torch.tensor(mask, dtype=torch.long).to(self.__device),
                )
                if self.__task == Task.TOKEN:
                    pred = torch.argmax(outputs.logits, dim=2)
                    prob = nn.functional.log_softmax(outputs.logits, dim=-1)
                    max_log_prob, _ = torch.max(prob, dim=-1)
                    total_log_prob = max_log_prob.sum(dim=1)
                    prob = total_log_prob / len(pred)
                else:
                    pred = torch.argmax(outputs.logits, dim=1).tolist()
                    prob = [
                        max(all_prob)
                        for all_prob in nn.functional.log_softmax(outputs.logits, dim=-1)
                    ]
            del outputs        
            gc.collect()
            torch.cuda.empty_cache()   
        return pred, prob

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

            optimizer.zero_grad()

            outputs = self.__model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # compute training accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(
                -1, self.__model.num_labels
            )  # shape (batch_size * seq_len, num_labels)

            loss = loss_fn(active_logits, flattened_targets)  # type: ignore

            tr_loss += loss.item()
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

            with torch.no_grad():
                # Move predictions and targets to CPU only once for accuracy calculation
                predictions = flattened_predictions.cpu().numpy()
                targets = flattened_targets.cpu().numpy()
                
                # Extend predictions and targets
                tr_preds.extend(predictions)
                tr_labels.extend(targets)
                
                # Compute the accuracy
                tmp_tr_accuracy = accuracy_score(predictions, targets)
                tr_accuracy += tmp_tr_accuracy

            # TODO:
            #torch.nn.utils.clip_grad_norm_(
            #    parameters=self.__model.parameters(), max_norm=10
            #)

            # backward pass
            loss.backward()
            optimizer.step()
            # scheduler.step()
            progress_bar.update(1)

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        acc = accuracy_score(tr_labels, tr_preds)

        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

        print(f"Overall acc: {acc}")
        print(f"Overall loss: {tr_loss}")

        return tr_loss, acc

    def __valid(self, testing_loader, loss_fn, id2label, end=False):
        # put model in evaluation mode
        self.__model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(testing_loader):
                ids = batch["ids"].to(self.__device, dtype=torch.long)
                mask = batch["mask"].to(self.__device, dtype=torch.long)
                targets = batch["targets"].to(self.__device, dtype=torch.long)

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

                eval_labels.extend(targets.cpu())
                eval_preds.extend(predictions.cpu())

                tmp_eval_accuracy = accuracy_score(
                    targets.cpu().numpy(), predictions.cpu().numpy()
                )
                eval_accuracy += tmp_eval_accuracy
                
                del outputs
                torch.cuda.empty_cache()

            labels = [id2label[id.item()] for id in eval_labels]
            predictions = [id2label[id.item()] for id in eval_preds]

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_steps

            acc = accuracy_score(eval_labels, eval_preds)

            print(f"Validation Loss: {eval_loss}")
            print(f"Validation Accuracy: {eval_accuracy}")
            print(f"Overall accuracy: {acc}")
            print(f"Overall loss: {eval_loss}")

            if end:
                return labels, predictions
            
            report = self.__util.validate_report(labels, predictions, output=True)
            return eval_loss, acc, report['macro avg']['f1-score'], report['weighted avg']['f1-score']

    def forward(self, x):
        return self.__model(x)
