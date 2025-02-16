# INFO: Baseline model

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def c_overlap_c(text):
    return "OVERLAP" if (" og " in text and len(text) < 20) or " også " in text or ("," in text and len(text) < 5) or text.strip() == '' else "O"

def t_overlap_d(text):
    return "OVERLAP" if ((" på " in text or " i " in text or " den ") and len(text) < 20) or text.strip() == '' else "O"

def t_before_d(text):
    return "BEFORE" if " før " in text or " for " in text else "O"

def t_before_c(text):
    return "BEFORE" if " før " in text or " for " in text or " av " in text else "O"

def t_before_t(text):
    return "BEFORE" if " etter " in text else "O"
    
RULES = {
    "CONDITION_OVERLAP_CONDITION": c_overlap_c,
    "TREATMENT_OVERLAP_DATE": t_overlap_d,
    "TREATMENT_BEFORE_DATE": t_before_d,
    "TREATMENT_BEFORE_CONDITION": t_before_c,
    "TREATMENT_BEFORE_TREATMENT": t_before_t,
}


class Baseline:
    def __init__(self):
        pass
        
    def run(self, e_i, e_j):
    
        sentences = sent_tokenize(e_i['Context'])

        pred = "O"
        for sentence in sentences:
            if e_i['Text'] in sentence and e_j['Text'] in sentence:
                e_i_idx = sentence.index(e_i['Text']) + len(e_i['Text'])
                e_j_idx = sentence.index(e_j['Text'])
                
                if e_j_idx > e_i_idx:
                    first = True
                    start_idx = e_i_idx
                    end_idx = e_j_idx
                else:
                    first = False
                    start_idx = e_j_idx
                    end_idx = e_i_idx
                
                text_between = sentence[start_idx:end_idx]
                
                if e_i['MedicalEntity'] == "CONDITION" and e_j['MedicalEntity'] == "CONDITION":
                    pred = RULES["CONDITION_OVERLAP_CONDITION"](text_between)
                elif e_i['MedicalEntity'] == 'TREATMENT' and (e_j['TIMEX'] is not None or e_j['TIMEX'] != ''):
                    pred1 = RULES["TREATMENT_BEFORE_DATE"](text_between) if first else "O"
                    pred2 = RULES["TREATMENT_OVERLAP_DATE"](text_between)
                    return pred1 if pred1 != 'O' else pred2
                elif e_i['MedicalEntity'] == 'TREATMENT' and e_j['MedicalEntity'] == "CONDITION":
                    pred = RULES['TREATMENT_BEFORE_CONDITION'](text_between) if first else "O"
                elif e_i['MedicalEntity'] == "TREATMENT" and e_j["MedicalEntity"] == "TREATMENT":
                    pred = RULES["TREATMENT_BEFORE_TREATMENT"](text_between) if not first else "O"
                    
                # print(f"####### \n e_i: {e_i['Text']}, e_j: {e_j['Text']}, pred: {pred} \nText: {text_between} ")
                    
        return pred



if __name__ == "__main__":
    pass
    