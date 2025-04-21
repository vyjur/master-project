# INFO: Baseline model

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


def c_overlap_c(text, text_wo):
    return (
        "OVERLAP"
        if (" og " in text and len(text) < 20)
        or " også " in text
        or ("," in text and len(text_wo) < 5)
        or text_wo.strip() == ""
        else "O"
    )


def t_overlap_d(text, text_wo):
    return (
        "OVERLAP"
        if (
            (
                " på " in text
                or " i " in text
                or " den " in text
                or " pga " in text
                or " tilbake " in text
                or " pga. "
            )
            and len(text_wo) < 30
        )
        or text_wo.strip() == ""
        else "O"
    )


def t_before_d(text, text_wo):
    return "BEFORE" if " før " in text or " for " in text else "O"


def t_before_c(text, text_wo):
    return "BEFORE" if " før " in text or " for " in text or " av " in text else "O"


def t_before_t(text, text_wo):
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

    def run(self, e_i, e_j, output=False):
        sentences = sent_tokenize(e_i["Context"])

        pred = "O"
        for sentence in sentences:
            if e_i["Text"] in sentence and e_j["Text"] in sentence:
                e_i_idx = sentence.index(e_i["Text"])
                e_j_idx = sentence.index(e_j["Text"])

                if e_j_idx > e_i_idx:
                    first = True
                    start_idx = e_i_idx
                    end_idx = e_j_idx + len(e_j["Text"])
                else:
                    first = False
                    start_idx = e_j_idx
                    end_idx = e_i_idx + len(e_i["Text"])

                text_between = sentence[start_idx:end_idx]
                text_without = (
                    text_between.replace(e_i["Text"], "")
                    .replace(e_j["Text"], "")
                    .replace("\n", " ")
                    .strip()
                )
                if output:
                    print("Sentence:", sentence)
                    print(
                        f"1:{e_i['Text']} -----", text_between, f"------2:{e_j['Text']}"
                    )
                    print(e_i_idx, e_j_idx)
                    print(start_idx, end_idx)
                    print(text_without, print(len(text_without)))

                if (
                    e_i["MedicalEntity"] == "CONDITION"
                    and e_j["MedicalEntity"] == "CONDITION"
                ):
                    pred = RULES["CONDITION_OVERLAP_CONDITION"](
                        text_between, text_without
                    )
                elif e_j["MedicalEntity"] in ["CONDITION", "TREATMENT"] and (
                    e_i["TIMEX"] is not None or e_i["TIMEX"] != ""
                ):
                    # pred1 = RULES["TREATMENT_BEFORE_DATE"](text_between) if first else "O"
                    pred2 = RULES["TREATMENT_OVERLAP_DATE"](text_between, text_without)
                    # if output:
                    # print(pred1, pred2)
                    # return pred1 if pred1 != 'O' else pred2
                    pred = pred2
                elif e_i["MedicalEntity"] in ["CONDITION", "TREATMENT"] and (
                    e_j["TIMEX"] is not None or e_j["TIMEX"] != ""
                ):
                    # pred1 = RULES["TREATMENT_BEFORE_DATE"](text_between) if first else "O"
                    pred2 = RULES["TREATMENT_OVERLAP_DATE"](text_between, text_without)

                    if output:
                        print(" i " in text_between)

                        print(len(text_without))
                    # if output:
                    # print(pred1, pred2)
                    # return pred1 if pred1 != 'O' else pred2
                    pred = pred2
                elif (
                    e_i["MedicalEntity"] == "TREATMENT"
                    and e_j["MedicalEntity"] == "CONDITION"
                ):
                    pred = (
                        RULES["TREATMENT_BEFORE_CONDITION"](text_between, text_without)
                        if first
                        else "O"
                    )
                elif (
                    e_i["MedicalEntity"] == "TREATMENT"
                    and e_j["MedicalEntity"] == "TREATMENT"
                ):
                    pred = (
                        RULES["TREATMENT_BEFORE_TREATMENT"](text_between, text_without)
                        if not first
                        else "O"
                    )

                # print(f"####### \n e_i: {e_i['Text']}, e_j: {e_j['Text']}, pred: {pred} \nText: {text_between} ")

                break
        # Info: we are only doing two categories, so disregarding the other rules.
        if pred == "OVERLAP":
            return pred
        return "O"


if __name__ == "__main__":
    pass

