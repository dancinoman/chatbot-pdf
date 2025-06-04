
# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
from crewai import LLM

import os
import streamlit as st
import sys

sys.setrecursionlimit(5000)
# Loads model variables
os.environ["GROQ_API_KEY"] =st.secrets['general']['GROQ_API_KEY']

my_llm = LLM(
    api_key=os.getenv("GROQ_API_KEY"),
    model="groq/llama-3.3-70b-versatile"
)

# ## Agents

# ### First agent
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
	verbose=True,
    llm=my_llm
)


# ### Second agent
writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. ",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)


# ### Third agent
editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post ",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)


# ## Tasks
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, ",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in fully markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A fully text markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)

# ## Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=0
)

result = crew.kickoff(inputs={"topic": "step mother stuck in dryer machine"})


from IPython.display import Markdown
Markdown(result.raw)


# In[ ]:
