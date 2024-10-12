import torch
from transformers import pipeline
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString, Tag
from transformers import AutoTokenizer

PROMPTS = {
    1: '''### Task
        Your task is to generate an HTML version of an input text, marking up specific entities related to healthcare. The entities to be identified are: 'medical problems', 'treatments', and 'tests'. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of the entity.

        ### Entity Markup Guide
        {}
        Leave the text as it is if no such entities are found.

        ### Input Text: {}
        ### Output Text:
    '''
}

DUMMY_OUTPUT = [{'generated_text': '### Task\nYour task is to generate an HTML version of an input text, marking up specific entities related to healthcare. The entities to be identified are: \'medical problems\', \'treatments\', and \'tests\'. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of the entity.\n\n### Entity Markup Guide\nUse <span class="problem"> to denote a medical problem.\nUse <span class="treatment"> to denote a treatment.\nUse <span class="test"> to denote a test.\nLeave the text as it is if no such entities are found.\n\n### Input Text: While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers\' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.[92]\n\nDiosmectite, a natural aluminomagnesium silicate clay, is effective in alleviating symptoms of acute diarrhea in children,[93] and also has some effects in chronic functional diarrhea, radiation-induced diarrhea, and chemotherapy-induced diarrhea.[45] Another absorbent agent used for the treatment of mild diarrhea is kaopectate.\n\nRacecadotril an antisecretory medication may be used to treat diarrhea in children and adults.[86] It has better tolerability than loperamide, as it causes less constipation and flatulence.[94]\n### Output Text:\nWhile bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with <span class="problem">travelers\' diarrhea</span>, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.[92]\n\nDiosmectite, a natural aluminomagnesium silicate clay, is effective in alleviating symptoms of <span class="problem">acute diarrhea</span> in children,[93] and also has some effects in <span class="problem">chronic functional diarrhea</span>, <span class="problem">radiation-induced diarrhea</span>, and <span class="problem">chemotherapy-induced diarrhea</span>. Another absorbent agent used for the treatment of mild <span class="problem">diarrhea</span> is kaopectate.\n\n<span class="treatment">Racecadotril</span> an antisecretory medication may be used to treat <span class="problem">diarrhea</span> in children and adults.[86] It has better tolerability than loperamide, as it causes less <span class="problem">constipation</span> and <span class="problem">flatulence</span>.[94] ### Example Use Cases\n1.  **Medical Record**: A doctor wants to highlight a patient\'s medical problem, "diabetes," in their medical record. They use the HTML markup <span class="problem">diabetes</span> to denote the entity.\n2.  **Medical Research Article**: A researcher wants to highlight a treatment, "antibiotics," in their research article. They use the HTML markup <span class="treatment">antibiotics</span> to denote the entity.\n3.  **Healthcare Website**: A healthcare website wants to highlight a test, "blood pressure test," on their website. They use the HTML markup <span class="test">blood pressure test</span> to denote the entity. ### Code\n```html\n<span class="problem">travelers\' diarrhea</span>\n<span class="problem">acute diarrhea</span>\n<span class="problem">chronic functional diarrhea</span>\n<span class="problem">radiation-induced diarrhea</span>\n<span class="problem">chemotherapy-induced diarrhea</span>\n<span class="problem">diarrhea</span>\n<span class="treatment'}]

class LLM:
    
    def __init__(self, load: bool = True, dataset: list = [], tags_name: list = [], parameters: dict = [], align:bool = True):
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise fallback to CPU
        self.__pipeline = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device = device)    
        
        self.__tags_text = ''
        for tag in tags_name:
            self.__tags_text += f'Use <span class="{tag}"> to denote a {tag}.\n'
            
        # TODO: remove tokenizer from here?
        checkpoint = "distilbert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            

    def __process(self, input, output):
        text = output[0]['generated_text']
        start = text.find('Output Text:') + len('Output Text:')
        stop = text.find('### Example Use Cases')
        
        if stop == -1:
            stop = text.find('### Code')
            
        relevant_text = text[start:stop]
        relevant_text = relevant_text.replace('\n', '')
        
        soup = bs(relevant_text, "html.parser")
        tokens = []        
        bio_format = []
                
        max_length = len(list(soup.children))
    
        for i, child in enumerate(soup.children):
            if isinstance(child, NavigableString):
                tokenized = self.tokenizer(str(child), max_length=512, truncation=True, return_offsets_mapping=True)    
                curr_tokens = tokenized.tokens()
                if i == 0:
                    curr_tokens = curr_tokens[:len(curr_tokens)-1]
                elif i == max_length-1:
                    curr_tokens = curr_tokens[1:len(curr_tokens)]
                else:
                    curr_tokens = curr_tokens[1:len(curr_tokens)-1]
                tokens.extend(curr_tokens)
        
                for _ in curr_tokens:
                    bio_format.append('O')
            elif isinstance(child, Tag):
                tokenized = self.tokenizer(child.get_text(), max_length=512, truncation=True, return_offsets_mapping=True)    
                curr_tokens = tokenized.tokens()
                
                if i == 0:
                    curr_tokens = curr_tokens[:len(curr_tokens)-1]
                elif i == max_length-1:
                    curr_tokens = curr_tokens[1:len(curr_tokens)]
                else:
                    curr_tokens = curr_tokens[1:len(curr_tokens)-1]
                curr_class = child['class'][0]
                tokens.extend(curr_tokens)
                for i, _ in enumerate(curr_tokens):
                    if i == 0:
                        bio_format.append(f"B-{curr_class}")
                    else:
                        bio_format.append(f"I-{curr_class}")
        test = self.tokenizer(input, max_length=512, truncation=True, return_offsets_mapping=True)    
        for i, tok in enumerate(test.tokens()):
            if tok != tokens[i]:
                tokens.insert(i, tok)   
        return bio_format
    
    def predict(self, data):
        all_outputs = []
        for val in data:
            # output = DUMMY_OUTPUT
            output = self.__pipeline(PROMPTS[1].format(self.__tags_text, val), max_new_tokens=512)
            processed_output = self.__process(val, output)
            all_outputs.append(processed_output)
        return all_outputs 

    
if __name__ == '__main__':
    import json
    
    with open('./data/Corona2.json') as f:
        d = json.load(f)

    dataset_sample = []

    for example in d['examples']:
        
        entities = [ (annot['start'], annot['end'], annot['value'], annot['tag_name']) for annot in example['annotations']]
        
        dataset_sample.append({
            'text': example['content'],
            'entities': entities
        })

    tags = set()

    for example in d['examples']:
        for annot in example['annotations']:
            tags.add(annot['tag_name'])
            
    tags = list(tags)

    model = LLM(True, dataset_sample, tags)
    pred1 = model.predict([dataset_sample[0]['text']])
    pred2 = model.predict([dataset_sample[0]['text'], dataset_sample[1]['text']])
    print(pred1)
    print(pred2)

    
