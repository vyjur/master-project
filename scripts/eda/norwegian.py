import os
import pypdf
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_languages(
    Language.BOKMAL, Language.NYNORSK
).build()

files = []
raw_files = os.listdir('./data/helsearkiv/journal')
annotated_files = os.listdir('./data/helsearkiv/annotated')
annotated_files = [file.replace('.pdf', '') for file in annotated_files]

files = [file for file in raw_files if file.replace('.pdf', '') not in annotated_files]

al_data = []

norwegian = {
    'bokmaal': 0,
    'nynorsk': 0,
    'none': 0
}

for i, doc in enumerate(files):
    reader = pypdf.PdfReader('./data/helsearkiv/journal/' + doc)
    for j, page in enumerate(reader.pages):
        detected_language = detector.detect_language_of(page.extract_text())
        if detected_language == Language.BOKMAL:
            norwegian['bokmaal'] += 1
        elif detected_language == Language.NYNORSK:
            norwegian['nynorsk'] += 1
        else:
            norwegian['none'] += 1
       
print(norwegian) 



