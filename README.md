from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import streamlit as st

# Load the text data
file_path = '/content/infoo - Sheet1.csv'
text_data = pd.read_csv(file_path)

# Initialize the OpenAI model
llm = OpenAI(api_key='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')
# Define the prompt template for health statistics data
prompt_template = PromptTemplate(
    input_variables=["data_description", "question"],
    template="""
    you are a chatbot that gives information on natural disasters you have acces to the following data:
    {data_description}

    Question: {question}

    Please provide a detailed answer based on the data.
    """
)

# Create the LangChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate data description for the health statistics
def generate_data_description():
    sample_entries = text_data.sample(min(len(text_data), 5))  # Get up to 5 random rows
    description = "The dataset contains various health statistics over different years. It includes data points such as Infant Mortality Rate, Life Expectancy, Maternal Mortality Rate, Prevalence of Diabetes, and Prevalence of Hypertension. The data is structured with the following columns:\n"
    description += "-Keyword: The word to trigger a response.\n"
    description += "-Response: The respo nse\n"
    description += "Example entries:\n"
    description += "\n".join(f"Keyword: {row['Keyword']}, Response: {row['Response']}" for _, row in sample_entries.iterrows())
    return description

def get_response(question):
    data_description = generate_data_description()
    response = chain.run(data_description=data_description, question=question)
    return response

# Allow for dynamic input via user promp
if __name__ == "__main__":
    while True:
        user_question = input("Please enter your question or type 'exit' to quit: ")
        if user_question.lower() == 'exit':
            break
        answer = get_response(user_question)
        print("Answer:", answer)
