from transformers import AutoModel, AutoTokenizer, pipeline
from pdfquery import PDFQuery
import warnings

# Load Legal-BERT for encoding
legal_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", clean_up_tokenization_spaces=True)
legal_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

warnings.filterwarnings('ignore')
# Load DistilBERT for QA
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device='cuda:0')

# Your legal article or document
def get_from_pdf(pdf_location):
    pdf_file = PDFQuery(pdf_location)
    pdf_file.load()
    text_elements = pdf_file.pq('LTTextLineHorizontal')
    return ''.join([t.text for t in text_elements if t.text.strip() != '' and t.text.strip().replace('_','') != ""])

context = get_from_pdf('fine-tune/train/Contract_of_PurchaseSale.pdf')
print(f'\n{context}\n')

# Preprocess and encode using Legal-BERT
inputs = legal_tokenizer(context, return_tensors="pt", truncation=True, padding=True)
legal_embeddings = legal_model(**inputs)

# Question answering on the encoded text
question = "When the docuemnt were made?"
result = qa_pipeline(question=question, context=context)

# Output the result
print('Question: ' + question)
print(result)
