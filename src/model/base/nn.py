import os
import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader
from preprocess.setup import Preprocess
from sklearn.metrics import accuracy_score
from model.regularization.early_stopping import EarlyStopping
from model.util import Util
from structure.enum import Task
import wandb
from model.tuning.setup import TuningConfig
import numpy as np

sweep_config = TuningConfig.get_config()

import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class NN(nn.Module):
    def __init__(
        self,
        model: nn.Module,
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
        super(NN, self).__init__()
        self.__device = "cuda:0" if cuda.is_available() else "cpu"

        if self.__device != "cpu":
            torch.cuda.set_device(self.__device)

        self.device = self.__device

        print("Using:", self.__device, "with NN")

        self.tokenizer = tokenizer
        self.__task = task
        self.__base_model = model
        self.__save = save
        self.__util = util if util is not None else Util()
        

        self.__project_name = project_name
        self.__tags_name = tags_name
        self.__dataset = dataset

        embedding_dim = parameters["embedding_dim"]

        self.__processed = {}

        if load:
            tag_to_ix, ix_to_tag = self.__util.get_tags(task, tags_name)

            if task == Task.TOKEN:
                START_ID = max(ix_to_tag.keys()) + 1
                STOP_ID = max(ix_to_tag.keys()) + 2

                tag_to_ix[START_TAG] = START_ID
                tag_to_ix[STOP_TAG] = STOP_ID

            self.__processed["label2id"] = tag_to_ix
            self.__processed["id2label"] = ix_to_tag
        
        self.__vocab_size = self.tokenizer.vocab_size  # type: ignore
        hidden_dim = parameters["max_length"]

        if load:
            self.__model = model(
                1, self.__vocab_size, tag_to_ix, embedding_dim, hidden_dim
            )
            self.__model.load_state_dict(
                torch.load(save + "/model.pth", weights_only=False)
            )
        else:
            tune = parameters["tune"]
            if tune:
                sweep_config["parameters"]["valid_batch_size"] = {
                    "value": parameters["valid_batch_size"]
                }
                sweep_config["parameters"]["shuffle"] = {"value": parameters["shuffle"]}
                sweep_config["parameters"]["num_workers"] = {
                    "value": parameters["num_workers"]
                }
                sweep_id = wandb.sweep(
                    sweep_config,
                    project=f"{project_name}-{task}-nn-model".replace('"', ""),
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
                    "embedding_dim": parameters["embedding_dim"],
                    "max_length": parameters["max_length"],
                    "stride": parameters["stride"],
                    "shuffle": parameters["shuffle"],
                    "num_workers": parameters["num_workers"],
                    "evaluation_strategy": "epoch",
                    "save_strategy": "epoch",
                    "logging_strategy": "epoch",
                    "tune": tune,
                }
                self.train(wandb.config)

    def train(self, config=None):
        
        with wandb.init(project=f"{self.__project_name}-{self.__task}-nn-model".replace('"', "")):  # type: ignore

            if config is None:
                config = wandb.config

            print(config)
            
            self.__processed = Preprocess(
                self.tokenizer, config["max_length"], config['stride'], self.__util
            ).run_train_test_split(self.__task, self.__dataset, self.__tags_name)
            self.__class_weights = self.__util.class_weights(
                self.__task, self.__processed["dataset"], self.__device
            )

            if self.__task == Task.TOKEN:
                START_ID = max(self.__processed["id2label"].keys()) + 1
                STOP_ID = max(self.__processed["id2label"].keys()) + 2

                self.__processed["label2id"][START_TAG] = START_ID
                self.__processed["label2id"][STOP_TAG] = STOP_ID

            train_params = {
                "batch_size": config["batch_size"],
                "shuffle": config["shuffle"],
                "num_workers": config["num_workers"],
            }

            test_params = {
                "batch_size": config["valid_batch_size"],
                "shuffle": config["shuffle"],
                "num_workers": config["num_workers"],
            }

            training_loader = DataLoader(self.__processed["train"], **train_params)
            valid_loader = DataLoader(self.__processed["valid"], **test_params)
            testing_loader = DataLoader(self.__processed["test"], **test_params)
            
            if hasattr(self, '__model'):
                del self.__model
                torch.cuda.empty_cache()

            self.__model = self.__base_model(
                config["batch_size"],
                self.__vocab_size,
                self.__processed["label2id"],
                config["embedding_dim"],
                config["max_length"],
            ).to(self.__device)
            
            self.__model.num_labels = len(self.__processed["id2label"])

            loss_fn = nn.CrossEntropyLoss(weight=self.__class_weights)
            optimizer = torch.optim.Adam(  # type: ignore
                self.__model.parameters(),
                lr=config["learning_rate"],
                weight_decay=1e-4,
            )

            early_stopping = EarlyStopping(patience=5, delta=0.01)
            for t in range(config["epochs"]):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loss, train_acc = self.__train(
                    training_loader, loss_fn, optimizer
                )
                val_loss, val_acc = self.__valid(
                    valid_loader, loss_fn, self.__processed["id2label"]
                )
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
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
            if not config["tune"]:
                if not os.path.exists(self.__save):
                    os.makedirs(self.__save)
                torch.save(self.__model.state_dict(), self.__save + "/model.pth")

    def predict(self, data, pipeline=False):
        data_tensor = torch.tensor(data, dtype=torch.long).to(self.__device)
        self.__model.batch = data_tensor.shape[0]
        outputs = self.__model(data_tensor)
        if self.__task == Task.TOKEN:
            return outputs[0], outputs[1]
        else:
            pred = torch.argmax(outputs, axis=1).tolist()  # type: ignore
            prob = [
                max(all_prob)  # type: ignore
                for all_prob in nn.functional.log_softmax(
                    outputs, dim=-1
                )  # type:ignore
            ]

            return pred, prob  # type: ignore

    def __train(self, training_loader, loss_fn, optimizer):
        self.__model.train()
        tr_loss = 0

        all_preds, all_targets = [], []
        for idx, batch in enumerate(training_loader):
            ids = batch["ids"].to(self.__device, dtype=torch.long)
            targets = batch["targets"].to(self.__device, dtype=torch.long)
            self.__model.batch = ids.shape[0]

            flattened_targets = targets.view(-1).cpu().numpy()

            optimizer.zero_grad()

            if self.__task == Task.TOKEN:
                outputs, _ = self.__model(ids)
                loss = self.__model.neg_log_likelihood(ids, targets)
                predictions = outputs.view(-1).cpu().numpy()
            else:
                outputs = self.__model(ids)
                loss = loss_fn(outputs, targets.view(-1))
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(predictions)
            all_targets.extend(flattened_targets)
            tr_loss += loss.item()

            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Batch {idx}, Loss: {loss.item()}")

        acc = accuracy_score(all_targets, all_preds)

        return tr_loss, acc

    def __valid(self, testing_loader, loss_fn, id2label, end=False):
        # put model in evaluation mode
        self.__model.eval()

        eval_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for _, batch in enumerate(testing_loader):
                ids = batch["ids"].to(self.__device, dtype=torch.long)
                targets = batch["targets"].to(self.__device, dtype=torch.long)
                self.__model.batch = ids.shape[0]

                if self.__task == Task.TOKEN:
                    outputs, _ = self.__model(ids)
                    loss = self.__model.neg_log_likelihood(ids, targets)
                    predictions = outputs.view(-1).cpu().numpy()
                else:
                    outputs = self.__model(ids)
                    loss = loss_fn(outputs, targets.view(-1))
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                flattened_targets = targets.view(-1).cpu().numpy()
                all_preds.extend(predictions)
                all_targets.extend(flattened_targets)
                eval_loss += loss.item()

            acc = accuracy_score(all_targets, all_preds)

            if end:
                labels = [id2label[id] for id in all_targets]
                predictions = [id2label[id] for id in all_preds]

                print(f"Validation Accuracy: {acc}")

                return labels, predictions

            return eval_loss, acc

    def forward(self, x):
        self.__model.batch = x.shape[0]
        return self.__model(x)
