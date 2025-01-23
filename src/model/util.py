import torch
from sklearn.metrics import classification_report
from structure.enum import Task

class Util:
    
    def __init__(self):
        pass
    
    def validate_report(self, labels, predictions):
        
        print("### BIO-Scheme")
        print(classification_report(labels, predictions))

        cat_labels = [ lab.replace("B-", "").replace("I-", "") for lab in labels]
        cat_predictions = [ lab.replace("B-", "").replace("I-", "") for lab in predictions]
        
        tags = list(set(cat_labels).union(set(cat_predictions)))

        print("### Summary")
        print(classification_report(cat_labels, cat_predictions, labels=tags)) 
        
    def get_tags(self, task, tags_name, default=True):
        tags = set()
        
        
        for tag in tags_name:
            if tag == "O":
                continue
                
            if task == Task.TOKEN:
                tags.add(f"B-{tag}")
                tags.add(f"I-{tag}")
            else:
                tags.add(tag)
                
        tags = list(tags)

        if default:
            label2id = {k: v+1 for v, k in enumerate(tags)}
            id2label = {v+1: k for v, k in enumerate(tags)}

            label2id['O'] = 0
            id2label[0]='O'
        else:
            label2id = {k: v for v, k in enumerate(tags)}
            id2label = {v: k for v, k in enumerate(tags)}
        
        return label2id, id2label
    
    def class_weights(self, task, dataset, device):
        word_count = {}

        for ex in dataset:
            if task == Task.TOKEN:
                for word in ex['labels']:
                    # Assuming 'word' is a string
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
            else:
                if ex['labels'] in word_count:
                        word_count[ex['labels']] += 1
                else:
                    word_count[ex['labels']] = 1

        print(word_count)

        total_samples = sum(word_count.values())

        # Calculate class weights
        class_weights = {class_label: total_samples / (len(word_count) * count) 
                        for class_label, count in word_count.items()}

        # Display the class weights
        print(class_weights)
        class_weights = torch.tensor(list(class_weights.values())).to(device)
        return class_weights
