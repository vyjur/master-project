import torch
from typing import List
from sklearn.metrics import classification_report
from structure.enum import Task, NER_SCHEMA


class Util:
    def __init__(self, schema: NER_SCHEMA = None):
        self.schema = schema

    def validate_report(self, labels, predictions, output=True):
        if self.schema is not None:
            print(f"### {self.schema.name}-Scheme")
        else:
            print("### Summary")

        tags = list(set(labels).union(set(predictions)))
        report = classification_report(labels, predictions, output_dict=True)
        print(classification_report(labels, predictions))

        print("### Without O")
        try:
            tags.remove("O")
        except:
            pass

        try:
            tags.remove("BEFOREOVERLAP")
        except:
            pass

        if output:
            print(classification_report(labels, predictions, labels=tags))

        if self.schema is not None:
            cat_labels = [self.remove_schema(lab) for lab in labels]
            cat_predictions = [self.remove_schema(lab) for lab in predictions]

            tags = list(set(cat_labels).union(set(cat_predictions)))

            report = classification_report(
                cat_labels, cat_predictions, labels=tags, output_dict=True
            )

            if output:
                print("### Summary")
                print(classification_report(cat_labels, cat_predictions, labels=tags))

            try:
                tags.remove("O")
                if output:
                    print("### Without O")
                    print(
                        classification_report(cat_labels, cat_predictions, labels=tags)
                    )
            except:
                pass

        return report

    def remove_schema(self, text: str):
        if self.schema == NER_SCHEMA.BIO:
            text = text.replace("B-", "").replace("I-", "")
        elif self.schema == NER_SCHEMA.IO:
            text = text.replace("I-", "")
        elif self.schema == NER_SCHEMA.IOE:
            text = text.replace("E-", "").replace("I-", "")
        return text

    def get_tags(self, task: Task, tags_name: List):
        tags = set()

        for tag in tags_name:
            if task == Task.TOKEN and tag != "O":
                if self.schema == NER_SCHEMA.BIO:
                    tags.add(f"B-{tag}")
                    tags.add(f"I-{tag}")
                elif self.schema == NER_SCHEMA.IO:
                    tags.add(f"I-{tag}")
                elif self.schema == NER_SCHEMA.IOE:
                    tags.add(f"E-{tag}")
                    tags.add(f"I-{tag}")
                else:
                    tags.add(tag)
            else:
                tags.add(tag)

        tags = list(tags)
        tags.sort()

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

        # Sort class weights alphabetically by class labels
        sorted_class_weights = dict(sorted(class_weights.items()))
        # sorted_class_weights = class_weights

        print(sorted_class_weights)
        # reduce = [1, 0.7, 1, 0.86, 2.409]

        # sorted_class_weights = {key: value * reduce[i] for i, (key, value) in enumerate(sorted_class_weights.items())}

        # Display the sorted class weights
        print(sorted_class_weights)

        # Convert to tensor
        class_weights_tensor = torch.tensor(list(sorted_class_weights.values())).to(
            device
        )

        return class_weights_tensor

    def tokens_mapping(self, tokenized, annot, word_length, terms):
        tokens_annot = []
        for i in range(len(tokenized)):
            if self.schema in {NER_SCHEMA.BIO, NER_SCHEMA.IO, NER_SCHEMA.IOE}:
                if tokenized.offsets[i] == (0, 0):
                    tokens_annot.append("O")
                    continue

                tag_prefix = ""
                if self.schema == NER_SCHEMA.BIO:
                    tag_prefix = "B-" if tokenized.offsets[i][0] == 0 else "I-"
                    if i != 0 and terms[i - 1] == terms[i]:
                        tag_prefix = "I-"
                elif self.schema == NER_SCHEMA.IOE:
                    tag_prefix = (
                        "E-" if tokenized.offsets[i][1] == word_length[i] else "I-"
                    )
                    if i != (len(tokenized) - 1) and terms[i + 1] == terms[i]:
                        tag_prefix = "I-"
                elif self.schema == NER_SCHEMA.IO:
                    tag_prefix = "I-"

                tokens_annot.append(
                    f"{tag_prefix}{annot[i]}" if annot[i] != "O" else "O"
                )

        return tokens_annot

