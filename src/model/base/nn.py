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
    ):
        super(NN, self).__init__()
        self.__device = "cuda:0" if cuda.is_available() else "cpu"

        if self.__device != "cpu":
            torch.cuda.set_device(self.__device)
            
        self.device = self.__device

        print("Using:", self.__device, "with NN")

        self.tokenizer = tokenizer
        self.__task = task

        # TODO:
        # embedding_dim = self.tokenizer.model_max_length
        embedding_dim = 300
        
        if not load:
            processed = Preprocess(
                self.tokenizer, parameters["max_length"]
            ).run_train_test_split(task, dataset, tags_name)
            class_weights = Util().class_weights(task, processed["dataset"], self.__device)

            tag_to_ix = processed["label2id"]
            processed["label2id"] = tag_to_ix
            ix_to_tag=processed["id2label"]

        else:
            tag_to_ix, ix_to_tag = Util().get_tags(task, tags_name)

        if task == Task.TOKEN:
            START_ID = max(ix_to_tag.keys()) + 1
            STOP_ID = max(ix_to_tag.keys()) + 2

            tag_to_ix[START_TAG] = START_ID
            tag_to_ix[STOP_TAG] = STOP_ID

        vocab_size = self.tokenizer.vocab_size  # type: ignore
        hidden_dim = parameters["max_length"]

        if load:
            self.__model = model(1, vocab_size, tag_to_ix, embedding_dim, hidden_dim)
            self.__model.load_state_dict(
                torch.load(save + "/model.pth", weights_only=False)
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
                "shuffle": parameters["shuffle"],
                "num_workers":parameters["num_workers"] ,
            }

            test_params = {
                "batch_size": parameters["valid_batch_size"],
                "shuffle": parameters["shuffle"],
                "num_workers": parameters["num_workers"],
            }

            training_loader = DataLoader(processed["train"], **train_params)
            valid_loader = DataLoader(processed["valid"], **test_params)
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

            early_stopping = EarlyStopping(patience=5, delta=0.01)
            for t in range(parameters["epochs"]):
                print(f"Epoch {t + 1}\n-------------------------------")
                loss, acc = self.__train(training_loader, loss_fn, optimizer)
                wandb.log({"loss": loss, "accuracy": acc})  # type: ignore
                early_stopping(loss, self.__model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            
            print("### Valid set performance:")
            labels, predictions = self.__valid(
                valid_loader, self.__device, processed["id2label"]
            )
            Util().validate_report(labels, predictions)


            print("### Test set performance:")
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
            return outputs[0], outputs[1]
        else:
            pred = torch.argmax(outputs, axis=1).tolist()  # type: ignore
            prob = [
                max(all_prob)  # type: ignore
                for all_prob in nn.functional.log_softmax(outputs, dim=-1)  # type:ignore
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

            optimizer.zero_grad()

            if self.__task == Task.TOKEN:
                loss = self.__model.neg_log_likelihood(ids, targets)
                with torch.no_grad():
                    outputs = self.__model(ids)
                    predictions = outputs.view(-1).cpu().numpy()
                    flattened_targets = targets.view(-1).cpu().numpy()
            else:
                outputs = self.__model(ids)
                flattened_targets = targets.view(-1).cpu().numpy()
                loss = loss_fn(outputs, targets.view(-1))
                with torch.no_grad():
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

    def forward(self, x):
        self.__model.batch = x.shape[0]
        return self.__model(x)