import torch
from typing import List
from sklearn.metrics import classification_report
from structure.enum import Task, NER_SCHEMA


class Util:
    def __init__(self, schema: NER_SCHEMA = None):
        self.schema = schema

    def validate_report(self, labels, predictions):
        print("### BIO-Scheme")
        print(classification_report(labels, predictions))

        # TODO: SCHEMA
        cat_labels = [lab.replace("B-", "").replace("I-", "") for lab in labels]
        cat_predictions = [
            lab.replace("B-", "").replace("I-", "") for lab in predictions
        ]

        tags = list(set(cat_labels).union(set(cat_predictions)))

        print("### Summary")
        print(classification_report(cat_labels, cat_predictions, labels=tags))

    def get_tags(self, task: Task, tags_name: List, schema: NER_SCHEMA=NER_SCHEMA.BIO):
        
        # TODO: SCHEMA
        tags = set()

        for tag in tags_name:
            if task == Task.TOKEN and tag != "O":
                if schema == NER_SCHEMA.BIO:
                    tags.add(f"B-{tag}")
                    tags.add(f"I-{tag}")
            else:
                tags.add(tag)

        tags = list(tags)

        label2id = {k: v for v, k in enumerate(tags)}
        id2label = {v: k for v, k in enumerate(tags)}

        return label2id, id2label

    def class_weights(self, task, dataset, device):
        word_count = {}

        for ex in dataset:
            if task == Task.TOKEN:
                for word in ex["labels"]:
                    # Assuming 'word' is a string
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
            else:
                if ex["labels"] in word_count:
                    word_count[ex["labels"]] += 1
                else:
                    word_count[ex["labels"]] = 1

        print(word_count)

        total_samples = sum(word_count.values())

        # Calculate class weights
        class_weights = {
            class_label: total_samples / (len(word_count) * count)
            for class_label, count in word_count.items()
        }

        # Display the class weights
        print(class_weights)
        class_weights = torch.tensor(list(class_weights.values())).to(device)
        return class_weights
