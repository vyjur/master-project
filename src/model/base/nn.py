import os
import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader
from preprocess.setup import Preprocess
from sklearn.metrics import accuracy_score
from model.util import Util
from structure.enum import Task
import wandb

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class NN:
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
    ):
        self.__device = "cuda:0" if cuda.is_available() else "cpu"
        
        if self.__device != "cpu":
            torch.cuda.set_device(self.__device)

        print("Using:", self.__device, "with NN")

        self.tokenizer = tokenizer
        self.__task = task

        # TODO:
        # embedding_dim = self.tokenizer.model_max_length
        embedding_dim = 300
        processed = Preprocess(
            self.tokenizer, parameters["max_length"]
        ).run_train_test_split(task, dataset, tags_name)
        class_weights = Util().class_weights(task, processed["dataset"], self.__device)

        tag_to_ix = processed["label2id"]
        
        if len(tag_to_ix) != len(class_weights):
            del tag_to_ix['O']
            for tag in tag_to_ix:
                tag_to_ix[tag] -= 1

        if task == Task.TOKEN:
            START_ID = max(processed["id2label"].keys()) + 1
            STOP_ID = max(processed["id2label"].keys()) + 2

            tag_to_ix[START_TAG] = START_ID
            tag_to_ix[STOP_TAG] = STOP_ID

        vocab_size = self.tokenizer.vocab_size  # type: ignore

        hidden_dim = parameters["max_length"]

        if load:
            self.__model = model(1, vocab_size, tag_to_ix, embedding_dim, hidden_dim)
            self.__model.load_state_dict(
                torch.load(save + "/model.pth", weights_only=True)
            )
        else:
            wandb.init(project=f"{project_name}-{task}-nn-model".replace('"', ""))  # type: ignore
            wandb.config = {
                "learning_rate": parameters["learning_rate"],
                "epochs": parameters["epochs"],
                "batch_size": parameters["train_batch_size"],
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "logging_strategy": "epoch",
            }

            train_params = {
                "batch_size": parameters["train_batch_size"],
                "shuffle": True,
                "num_workers": 0,
            }

            test_params = {
                "batch_size": parameters["valid_batch_size"],
                "shuffle": True,
                "num_workers": 0,
            }

            training_loader = DataLoader(processed["train"], **train_params)
            testing_loader = DataLoader(processed["test"], **test_params)

            self.__model = model(
                parameters["train_batch_size"],
                vocab_size,
                tag_to_ix,
                embedding_dim,
                hidden_dim,
            ).to(self.__device)

            self.__model.num_labels = len(processed["id2label"])

            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(  # type: ignore
                self.__model.parameters(),
                lr=parameters["learning_rate"],
                weight_decay=1e-4,
            )

            for t in range(parameters["epochs"]):
                print(f"Epoch {t + 1}\n-------------------------------")
                loss, acc = self.__train(training_loader, loss_fn, optimizer)
                wandb.log({"loss": loss.item(), "accuracy": acc})  # type: ignore

            labels, predictions = self.__valid(
                testing_loader, self.__device, processed["id2label"]
            )
            Util().validate_report(labels, predictions)

            if not os.path.exists(save):
                os.makedirs(save)
            torch.save(self.__model.state_dict(), save + "/model.pth")
            wandb.finish()

    def predict(self, data, pipeline=False):
        data_tensor = torch.tensor(data, dtype=torch.long).to(self.__device)
        self.__model.batch = data_tensor.shape[0]
        outputs = self.__model(data_tensor)
        if self.__task == Task.TOKEN:
            return outputs
        else:
            pred = torch.argmax(outputs, axis=1).tolist()  # type: ignore
            prob = [max(all_prob)for all_prob in nn.functional.softmax(outputs, dim=-1)]

            return pred, prob  # type: ignore
    def __train(self, training_loader, loss_fn, optimizer):
        self.__model.train()
        tr_loss, tr_accuracy = 0, 0
        for idx, batch in enumerate(training_loader):
            ids = batch["ids"].to(self.__device, dtype=torch.long)
            targets = batch["targets"].to(self.__device, dtype=torch.long)
            self.__model.batch = ids.shape[0]

            self.__model.zero_grad()
            if self.__task == Task.TOKEN:
                loss = self.__model.neg_log_likelihood(ids, targets)
                with torch.no_grad():
                    outputs = self.__model(ids)
                    flattened_targets = targets.view(-1)
                    # shape (batch_size * seq_len,)
                    curr_loss = loss
                    predictions = outputs
                    tr_accuracy += accuracy_score(
                        targets.view(-1).numpy(),
                        predictions.view(-1).numpy(),
                    )
            else:
                outputs = self.__model(ids)
                flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
                loss = loss_fn(outputs, flattened_targets)
                predictions = outputs
                with torch.no_grad():
                    curr_loss = loss
                    flattened_predictions = torch.argmax(
                        outputs,
                        axis=1,  # type: ignore
                    )
                    tr_accuracy += accuracy_score(
                        targets.view(-1).numpy(), flattened_predictions
                    )  # shape (batch_size * seq_len,)

            tr_loss += curr_loss

            optimizer.step()
            loss.backward()
            if idx % 100 == 0:
                print(f"Batch {idx}, Loss: {loss.item()}")

        return tr_loss, tr_accuracy

    def __valid(self, testing_loader, device, id2label):
        # put model in evaluation mode
        self.__model.eval()

        _, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for _, batch in enumerate(testing_loader):
                ids = batch["ids"].to(device, dtype=torch.long)
                # mask = batch['mask'].to(device, dtype = torch.long)
                targets = batch["targets"].to(device, dtype=torch.long)
                self.__model.batch = ids.shape[0]
                outputs = self.__model(ids)

                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)

                # compute evaluation accuracy
                flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
                flattened_predictions = outputs.view(
                    -1
                )  # shape (batch_size * seq_len,)

                if self.__task != Task.TOKEN:
                    flattened_predictions = torch.argmax(
                        outputs,
                        axis=1,  # type: ignore
                    )  # shape (batch_size * seq_len,)

                # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                eval_labels.extend(flattened_targets.tolist())
                eval_preds.extend(flattened_predictions.tolist())

                tmp_eval_accuracy = accuracy_score(
                    flattened_targets.cpu().numpy(), flattened_predictions.cpu().numpy()
                )
                eval_accuracy += tmp_eval_accuracy

        labels = [id2label[id] for id in eval_labels]
        predictions = [id2label[id] for id in eval_preds]

        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Accuracy: {eval_accuracy}")

        return labels, predictions
