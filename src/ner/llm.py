prompts = {
    1: '''### Task
        Your task is to generate an HTML version of an input text, marking up specific entities related to healthcare. The entities to be identified are: 'medical problems', 'treatments', and 'tests'. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of the entity.

        ### Entity Markup Guide
        Use <span class="problem"> to denote a medical problem.
        Use <span class="treatment"> to denote a treatment.
        Use <span class="test"> to denote a test.
        Leave the text as it is if no such entities are found.

        ### Input Text: {}
        ### Output Text:
    '''
}

class LLM:
    
    def __init__(self, load: bool = True, dataset: list = [], tags_name: list = [], parameters: dict = []):
        pass
    
    def predict(self, data):
        pass