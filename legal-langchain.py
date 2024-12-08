import os
from dotenv import load_dotenv

from pdfquery import PDFQuery
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub

load_dotenv()


def get_from_pdf(pdf_location):
    pdf_file = PDFQuery(pdf_location)
    pdf_file.load()
    text_elements = pdf_file.pq('LTTextLineHorizontal')
    # Cleaning the text
    return ''.join([t.text for t in text_elements if t.text.strip() != '' and t.text.strip().replace('_','') != ""])


#document = get_from_pdf('pdf-documents/Contract_of_PurchaseSale.pdf')
with open('pdf-documents/Contract.txt', 'r') as f:
    contract_text = f.read()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(contract_text)

# start embedding
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embeddings)
store = vector_store.add_texts(chunks)

prompt = hub.pull("rlm/rag-prompt")

user_question = input('Ask a question:')
message = prompt.invoke({'context':'contract document', 'question': user_question}).to_messages()

assert len(message) == 1
print(message[0].content)

#Continue the tutorial there https://python.langchain.com/docs/tutorials/rag/
