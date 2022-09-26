import sys

import easyocr
import spacy
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

import tensorflow as tf
from pdf_redaction import PdfRedaction


pdf = sys.argv[1]

#Instantiate easyocr
reader = easyocr.Reader(['en'], gpu=True)  


def load_models():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    name_detector   = pipeline("ner", model=model, tokenizer=tokenizer)

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    signature_detector = tf.saved_model.load('exported-models\my_model\saved_model')

    date_detector = spacy.load("en_core_web_sm")

    return name_detector, signature_detector, date_detector


print("Loading Models...")
name_detector, signature_detector, date_detector = load_models()
print("Done Loading")

# Instantiate the redaction class
redactor = PdfRedaction(
    signature_detector, name_detector, date_detector, reader
)

#Function to run all methods
def run_redaction():
    #Convert pdf to images
    redactor.pdf_to_image(pdf)

    #Redact and return a modified pdf
    redactor.redact_pdf()
    


if __name__ == '__main__':
    run_redaction()




