from langchain_community.chat_models import ChatAnthropic, ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = pipeline("text-generation",
                                   model=model_name,
                                   tokenizer=tokenizer,
                                   torch_dtype=torch.float16,
                                   trust_remote_code=True,
                                   is_decoder=True,
                                   device='cuda:0')


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


def get_from_pdf(pdf_location):
    pdf_file = PDFQuery(pdf_location)
    pdf_file.load()
    text_elements = pdf_file.pq('LTTextLineHorizontal')
    # Cleaning the text
    return ''.join([t.text for t in text_elements if t.text.strip() != '' and t.text.strip().replace('_','') != ""])


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
