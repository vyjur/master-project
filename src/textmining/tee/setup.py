from textmining.tee.heideltime.python_heideltime import Heideltime
import xml.etree.ElementTree as ET
import pandas as pd

class TEExtract:
    
    def __init__(self):
        self.__heideltime = Heideltime()
        self.__heideltime_parser.set_document_type('NEWS')
        self.__heideltime_parser.set_language('auto-norwegian')
    
    def __run(self, text):
        result = self.__heideltime.parse(text)
        
        root = ET.fromstring(result)
        timex_elements = root.findall('.//TIMEX3')

        for timex in timex_elements:
            tid = timex.get('tid')
            ttype = timex.get('type')
            value = timex.get('value')
            text = timex.text
            
            # Get the full text of the document
            full_text = "".join(root.itertext())
            
            # Get the 50-character window around this TIMEX3 element
            char_window = self.__get_char_window(full_text, text, window_size=50)
            
            return {
                'id': tid,
                'type': {ttype},
                'value': {value},
                'text': {text},
                'context': {char_window}
            }
            
    def run(self, data):
        return pd.DataFrame([self.__run(text) for text in data]) 
        
    def __get_char_window(self, full_text, timex_text, window_size=50):
        # Find the index of the TIMEX3 text in the full text
        start_index = full_text.find(timex_text)
        end_index = start_index + len(timex_text)
        
        # Get the character window before and after the TIMEX3 text
        before_window = full_text[max(0, start_index - window_size):start_index]  # 50 characters before TIMEX3
        after_window = full_text[end_index:end_index + window_size]  # 50 characters after TIMEX3
        
        # Combine the before and after windows with the TIMEX3 text itself in the middle
        window = before_window + timex_text + after_window
        return window
        
    