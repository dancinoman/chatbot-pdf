from typing_extensions import List, TypedDict

from pdfquery import PDFQuery
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

#from langchain_core.documents import Document # Uncomment for pdf usage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from langchain import hub

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

user_question = input('Ask anything:')
#message = prompt.invoke({'context':'contract document', 'question': user_question}).to_messages()

class State(TypedDict):
    question: str
    context: str
    answer: str

def retrieve(state: State):
    retrieve_text = vector_store.similarity_search(state['question'])
    return {'context': retrieve_text}

def generate(state: State):
    text_content = contract_text
    message = prompt.invoke({'question': state['question'], 'context': text_content})
    response = llm.invoke(message)
    return {'answer': response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": user_question})
print(f'Answer: {result["answer"]}')
#Continue the tutorial there https://python.langchain.com/docs/tutorials/rag/
