from sklearn.metrics import classification_report

class Util:
    
    def __init__(self):
        pass
    
    def validate_report(self, labels, predictions):
        
        print("### BIO-Scheme")
        print(classification_report(labels, predictions))

        cat_labels = [ lab.replace("B-", "").replace("I-", "") for lab in labels]
        cat_predictions = [ lab.replace("B-", "").replace("I-", "") for lab in predictions]

        print("### Summary")
        print(classification_report(cat_labels, cat_predictions)) 