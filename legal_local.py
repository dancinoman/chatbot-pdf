#from langchain_community.chat_models import ChatAnthropic, ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import llm
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import PyPDF2
from pdfquery import PDFQuery
import torch

load_dotenv()


class LegalExpert:
    def __init__(self):
        self.system_prompt = self.get_system_prompt()

        self.user_prompt = HumanMessagePromptTemplate.from_template("{question}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )


        # falcon model
        model_name = "tiiuae/falcon-11B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_4bit=True)
        self.falcon_llm = pipeline("text-generation",
                                   model=model_name,
                                   tokenizer=tokenizer,
                                   torch_dtype=torch.float16,
                                   trust_remote_code=True,
                                   device_map="auto")

        print(f'Model {model_name} is set.')
        # create llm pipeline for model
        #model_name = "google/flan-t5-xl"

        self.huggingface_llm = pipeline("text-generation", model=model_name, tokenizer=tokenizer, device_map='auto')
        print('Hugging face pipeline set.')
        #self.openai_gpt4_llm = ChatOpenAI(temperature=0, max_tokens=256)
        #self.chat = ChatAnthropic()

        self.chain = full_prompt_template | self.huggingface_llm


        #self.chain = llm.LLMChain(llm=self.huggingface_llm, prompt=full_prompt_template)

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
            language=language, context=context, question=question
        )

pdf_file_loc = "Legal documentation/Contract_of_PurchaseSale.pdf"

def retrieve_pdf_text(pdf_file_loc):

    pdf_file = PDFQuery("Legal documentation/Contract_of_PurchaseSale.pdf")
    pdf_file.load()
    text_elements = pdf_file.pq('LTTextLineHorizontal')
    return [t.text for t in text_elements]


# create a streamlit app
print("Starting Document Explainer (that does not give advice)")

machine_reader = LegalExpert()

# create a upload file widget for a pdf


language = input("1.French/n2.English/n")
question = input("Ask a question? ")
run = input("Run?(Y/N)")

# Ask user to run
if run == 'Y':
    # run the model
    legal_response = machine_reader.run_chain(
        language=language, context=retrieve_pdf_text(pdf_file_loc), question=question
    )
    #Output the answer
    print(f"legal_response: {legal_response}")
