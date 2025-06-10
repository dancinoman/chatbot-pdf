import os
import sys
import pysqlite3
import re
import fitz
import streamlit as st
#from IPython.display import Markdown

import tiktoken

# Warning control
import warnings
warnings.filterwarnings('ignore')

sys.modules["sqlite3"] = sys.modules["pysqlite3"]

from crewai import Agent, Task, Crew
from crewai import LLM

GROQ_API_KEY = st.secrets['general']['GROQ_API_KEY']
MODEL_USED = "groq/llama-3.3-70b-versatile"
FILE_LOCATION = "pdf-documents/Contract_of_PurchaseSale.pdf"

def clean_text(text):
    """
    Cleans the extracted text from unwanted characters and extra spaces.

    Args:
        text: The extracted raw text from the PDF.

    Returns:
        str: The cleaned text.
    """
    # Remove non-printable characters (including newlines, tabs, etc.)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-ASCII characters (optional, adjust as needed)

    # Additional cleaning steps (e.g., removing page numbers, unwanted symbols)
    text = text.replace('\n', ' ').replace('\r', '')  # Remove line breaks

    # Strip leading and trailing spaces
    text = text.strip()

    return text

def get_from_text(location):

    with open(location, 'r') as f:
       return f.read()

def get_from_pdf(pdf_file):
    """
    Extracts text from uploaded PDF using PyMuPDF.

    Args:
        pdf_file: A file object containing the uploaded PDF.

    Returns:
        str: The cleaned text extracted from the PDF.
    """
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        page_text = ""
        for page in doc:
            page_text += page.get_text("text")
        cleaned_text = clean_text(page_text)
        text += cleaned_text
        return text
    except Exception as e:
        st.error(e)
        st.error("Load document failed. Try another one please.")
        return None

def model_run(query, document):
    sys.setrecursionlimit(5000)
    # Loads model variables
    os.environ["GROQ_API_KEY"]=GROQ_API_KEY

    my_llm = LLM(
        api_key=os.getenv("GROQ_API_KEY"),
        model=MODEL_USED
    )

    # ## Agents

    # ### First agent
    researcher = Agent(
        role="Researcher",
        goal="Research document by engaging to understand the query: {query}",
        backstory="""You search the most relevant information
                about the query: {query}.
                You understand the context and intention"
                from the user to find the most relevant information"
                You bring as much information that is requested.""",
        allow_delegation=False,
        verbose=True,
        llm=my_llm
    )


    # ### Second agent
    writer = Agent(
        role="Content Writer",
        goal="Write content to be concise and understandable",
        backstory=
        """You look at the {query} if there is any format requested
                You're working on writing the response
                for the user.
                You focus on facts with opinion free.
                You make sure that the response is understandable"""
                ,
        allow_delegation=False,
        verbose=True,
        llm=my_llm
    )


    # ## Tasks
    seek = Task(
        description=(
            "1. Prioritize the information requested "
                "on {query}.\n"
            "2. Find this information in {document}\n"
            "3. Formulate the content found by few lines\n"
        ),
        expected_output="A comprehensive content, with all information",
        agent=researcher,
    )

    response = Task(
        description=(
            "1. Look at the {query} if there is any format requested.\n"
            "2. Write format requested otherwise few lines.\n"
            "3. Correct any grammar errors.\n"
        ),
        expected_output="Priority on the prompt instruction(s) otherwise maximum 3 sentences."
            "in fully markdown format",
        agent=writer,
    )

    # ## Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[seek, response],
        verbose=1
    )

    result = crew.kickoff(inputs={"query": query,
                                  "document" : document
    })

    return result

def estimate_tokens(text):
    """Estimates the number of tokens in a given text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def main():
    with open("pdf-documents/CHA23131 Call of Cthulhu 7th Edition Quick-Start Rules.pdf", "rb") as f:
        document_text = get_from_pdf(f)
    print(type(document_text))
    print(f"Welcome to RAG with this model { MODEL_USED }")
    print(f"He is going to investigate the document {FILE_LOCATION} to answer your question")
    print(f"Number of tokens approximately in the document: {estimate_tokens(document_text)}")
    user_question = input('Posez votre question:')



    result = model_run(user_question, document_text)

    print(f"From the model {MODEL_USED} here is the answer \n",result.raw)

if __name__ == "__main__":
    # run for test
    main()
