from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from pdfquery import PDFQuery
import warnings

# Ignore warnings from model
warnings.filterwarnings("ignore")

# Load the Legal-BERT model and tokenizer
model_name = "ThePixOne/SeconBERTa1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Initialize a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer,  device='cuda:0')

pdf_location = 'legal documentation/Intellectual Property Agreement.pdf'



# Example legal context and question
question = "My mom is on your lips?"

# Get the context from pdf document
context = get_from_pdf(pdf_location)

# Use the pipeline to answer the question
result = qa_pipeline(question=question, context=context)

#Print the result
print(result)
