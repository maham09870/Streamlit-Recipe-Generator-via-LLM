from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# from langchain.chains import llm
from langchain.chains import SequentialChain, LLMChain
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
import os
import json
load_dotenv()


gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

result = gemini.invoke("Hello Gemini, how are you?")
print(result.dict()) 

rephrase = PromptTemplate(
    input_variables = ["query"],
    template = "Rephrase the following query: {query}"
)

rephrase_chain = LLMChain(llm=gemini,
                          prompt = rephrase, output_key="rephrased_query")

answer = PromptTemplate(
    input_variables=["rephrased_query"],
    template = "Answer the following query: {rephrased_query}"
)
answer_chain = LLMChain(
    llm=gemini,
    prompt=answer,
    output_key="answer"
)
final_chain = SequentialChain(
    chains=[rephrase_chain, answer_chain],
    input_variables=["query"],
    output_variables=["rephrased_query", "answer"],
    verbose=True
)
response = final_chain.invoke({"query": "What is hack to make a perfect dishes?"})
print(response)           

# Define tools that wrap around the chains
tools = [
    Tool(
        name="Rephrase Tool",
        func=lambda q: rephrase_chain.run(query=q),
        description="Rephrases the given query into clearer language"
    ),
    Tool(
        name="Answer Tool",
        func=lambda q: answer_chain.run(rephrased_query=q),
        description="Answers the given question"
    )
]

# Initialize agent with Gemini
agent = initialize_agent(
    tools=tools,
    llm=gemini,
    agent="zero-shot-react-description", 
    verbose=True
)

print("\n--- Running via Agent ---")
agent.run("What is hack to make a perfect dishes?")
