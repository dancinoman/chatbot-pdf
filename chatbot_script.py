
# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
from crewai import LLM

import os
import sys
import re
import fitz
import streamlit as st


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
    os.environ["GROQ_API_KEY"] =st.secrets['general']['GROQ_API_KEY']

    my_llm = LLM(
        api_key=os.getenv("GROQ_API_KEY"),
        model="groq/llama-3.3-70b-versatile"
    )

    # ## Agents

    # ### First agent
    researcher = Agent(
        role="Researcher",
        goal="Research document by engaging to understand the query: {query}",
        backstory="You search the most relevant information"
                "about the query: {query}."
                "You understand the context and intention"
                "from the user to find the most relevant information"
                "You bring as much information that is requested.",
        allow_delegation=False,
        verbose=True,
        llm=my_llm
    )


    # ### Second agent
    writer = Agent(
        role="Content Writer",
        goal="Write content to be concise and understandable",
        backstory="You look at the {query} if there is any format requested"
                "You're working on writing the response "
                "for the user."
                "You focus on facts with opinion free."
                "You make sure that the response is understandable",
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
            "3. Formulate the content found concisely\n"
        ),
        expected_output="A comprehensive content, with all information",
        agent=researcher,
    )

    response = Task(
        description=(
            "1. Look at the {query} if there is any format requested.\n"
            "2. Write the format requested that override by default concise content.\n"
            "3. Correct any grammar errors.\n"
        ),
        expected_output="A well-written response for the user "
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


    from IPython.display import Markdown
    Markdown(result.raw)

def main():
    user_question = input('Posez votre question:')

    document_text = get_from_text('other-document/test_for_text.txt')

    result = model_run(user_question, document_text)

    print(result)

if __name__ == "__main__":
    # run for test
    main()