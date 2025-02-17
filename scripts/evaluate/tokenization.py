from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'ltg/norbert3-small', trust_remote_code=True
)

text = "TAG as es ee ae bs es ee be (TAG)en dato[/TAG] bare masse tkest her en annen [TAG]dato[/TAG] BLABLA"

print(tokenizer.tokenize(text))