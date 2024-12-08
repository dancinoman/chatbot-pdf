from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.indexes import VectorstoreIndexCreator
from transformers import AutoTokenizer, pipeline
import streamlit as st
from pdfquery import PDFQuery
import torch
### TODO: make this page work


class LegalExpert:
    def __init__(self):
        self.system_prompt = self.get_system_prompt()

        self.user_prompt = HumanMessagePromptTemplate.from_template("{question}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        # mistralai model
        model_name = "mistralai/Mistral-7B-v0.1"

        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", offload_folder="offload")
        self.llm = pipeline("text-generation",
                                   model=model_name,
                                   tokenizer=tokenizer,
                                   torch_dtype=torch.float16,
                                   trust_remote_code=True,
                                   is_decoder=True,
                                   device='cuda')


        # create llm pipeline for model
        self.chain = full_prompt_template | self.llm

    def get_system_prompt(self):
        system_prompt = """
        You are a Canadian Legal Expert.
        Under no circumstances do you give legal advice.

        You are adept at explaining the law in laymans terms, and you are able to provide context to legal questions.
        While you can add context outside of the provided context, please do not add any information that is not directly relevant to the question, or the provided context.
        You speak {language}.
        ### CONTEXT
        {context}
        ### END OF CONTEXT
        """

        return SystemMessagePromptTemplate.from_template(system_prompt)

    def run_chain(self, language, context, question):
        return self.chain.invoke(
            {'language':language,'context':context, 'question':question}
        )

def get_from_pdf_lang(pdf_location):
    """pdf_loader = PyPDFLoader(pdf_location)
    text = pdf_loader.load()
    print(torch.cuda.is_available())  # Should return True if GPU is accessible
    model_light = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_model = HuggingFaceEmbeddings(model_name=model_light, model_kwargs={'device':'cuda'})

    vector_store = FAISS.from_documents(documents=text, embedding=embedding_model)

    index = VectorstoreIndexCreator(vector_store=vector_store, embedding=embedding_model)

    return index.from_documents(documents=text)"""
    # Load PDF and extract text
    pdf_loader = PyPDFLoader(pdf_location)
    text = pdf_loader.load()

    # Debug: Check the structure of the loaded text
    print("Loaded text structure:", text)

    # Check if GPU is available
    print(torch.cuda.is_available())  # Should return True if GPU is accessible

    model_light = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_model = HuggingFaceEmbeddings(model_name=model_light, model_kwargs={'device': 'cuda'})

    # Prepare documents based on the loaded text structure
    documents = []
    if isinstance(text, list):
        for doc in text:
            if isinstance(doc, str):
                documents.append(Document(page_content=doc))
            elif hasattr(doc, 'page_content'):
                documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))
            else:
                print("Unknown document structure:", doc)
    else:
        print("Unexpected format for loaded text:", text)

    # Create a FAISS vector store using documents and embedding model
    vector_store = FAISS.from_documents(documents, embedding=embedding_model)

    # Create an index using the vector store and embedding model
    index_creator = VectorstoreIndexCreator(vectorstore=vector_store, embedding=embedding_model)

    # Index the documents
    index = index_creator.from_documents(documents)

    return index



def get_from_pdf(pdf_location):
    pdf_file = PDFQuery(pdf_location)
    pdf_file.load()
    text_elements = pdf_file.pq('LTTextLineHorizontal')
    # Cleaning the text
    return ''.join([t.text for t in text_elements if t.text.strip() != '' and t.text.strip().replace('_','') != ""])

def load_reader():
    # create a streamlit app
    #st.title("Document Explainer (that does not give advice)")

    #if "LegalExpert" not in st.session_state:
    #    st.session_state.LegalExpert = LegalExpert()
    legal_expert = LegalExpert()

    # create a upload file widget for a pdf
    #pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    context = get_from_pdf('pdf-documents/Intellectual Property Agreement.pdf')

    #st.session_state.context = None
    # if a pdf file is uploaded
    #if pdf_file:
        # retrieve the text from the pdf
    #   if "context" not in st.session_state:
    #      st.session_state.context = retrieve_pdf_text(pdf_file)

    # create a button that clears the context
    #if st.button("Clear context"):
    #    st.session_state.__delitem__("context")
    #    st.session_state.__delitem__("legal_response")

    # if there's context, proceed
    #if "context" in st.session_state:
        # create a dropdown widget for the language
    #    language = st.selectbox("Language", ["English", "Français"])
        # create a text input widget for a question
    #    question = st.text_input("Ask a question")

    language = 'English'
    question='When was made the document?'
        # create a button to run the model
    #   if st.button("Run"):
            # run the model
    #        legal_response = st.session_state.LegalExpert.run_chain(
    #            language=language, context=st.session_state.context, question=question
    #        )
    legal_response = legal_expert.run_chain(language=language, context=context, question=question)
    print(f"legal_response: {legal_response}")
    #        if "legal_response" not in st.session_state:
    #            st.session_state.legal_response = legal_response

    #        else:
    #            st.session_state.legal_response = legal_response

    # display the response
    #if "legal_response" in st.session_state:
    #    st.write(st.session_state.legal_response)

print(get_from_pdf_lang('pdf-documents/Intellectual Property Agreement.pdf'))
