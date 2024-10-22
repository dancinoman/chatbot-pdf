from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from pdfquery import PDFQuery
import warnings
import torch


# Load Legal-BERT for encoding
legal_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", clean_up_tokenization_spaces=True)
legal_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

#warnings.filterwarnings('ignore')
# Load DistilBERT for QA
#qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device='cuda:0')

qa_pipeline = pipeline(
    "text-generation",
    model='mistralai/Mistral-7B-v0.1',
    device='cuda:0',
    max_length=100,
    truncation=True
    )

hf = HuggingFacePipeline(pipeline=qa_pipeline)

# Your legal article or document
def get_from_pdf(pdf_location):
    pdf_file = PDFQuery(pdf_location)
    pdf_file.load()
    text_elements = pdf_file.pq('LTTextLineHorizontal')
    # Cleaning the text
    return ''.join([t.text for t in text_elements if t.text.strip() != '' and t.text.strip().replace('_','') != ""])

context = get_from_pdf('pdf-documents/Contract_of_PurchaseSale.pdf')
print(f'\n{context}\n')

# Preprocess and encode using Legal-BERT
inputs = legal_tokenizer(context, return_tensors="pt", truncation=True, padding=True)
legal_embeddings = legal_model(**inputs)


template = 'Question: {question}'
prompt = PromptTemplate(template=template)
chain = prompt | hf
# Question answering on the encoded text
question = "Hi?"

# Output the result
print(chain.invoke({'question':question}))
