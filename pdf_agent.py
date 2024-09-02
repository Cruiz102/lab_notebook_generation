# What I want to generate:


# I want to generate a small script that does the following


# Parse the Data from documentation once. Saved the data.
# Do the inference on the data.
# Add a LLM that uses that inference each time they ask a question.


#The script for parsing all the data inside the database should let you 
# decided where to saved the embeddings, how to parse the data.


# Do the Lammps documentation rag. Show the retrieved data.  
# Show from what page is the data.

# 
# How to implement it:
# Use OpenAI chat Completions
# Parse the data with PyPDF2
# Index the Data with Chroma
# Make a function that  makes as query and get the information
# The Query function should have the following:
# query by texts, n_results should be an hyperparameter
#  Use the Function calling of OpenAI
from openai import OpenAI
import openai
from tqdm import tqdm
from agent import Prompt

from PyPDF2 import PdfReader
import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
import chromadb
client = chromadb.PersistentClient(path="./")
client.heartbeat() 
def create_collection(collection_name: str,database_path,  pdf_file: str):
        client = chromadb.PersistentClient(path=database_path)
        collection = client.create_collection(collection_name)
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        for i in tqdm(range(num_pages)):
            collection.add(
            documents=[reader.pages[i].extract_text()], 
            metadatas=[{"source": "laamps_documentation", 'page':i }], # filter on these!
            ids=[f"laamps_page:${i}"], # unique for each doc
        )

def chroma_query(collection_name: str, database_path:str, query: str, n_results: int):
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_collection(collection_name)
    results = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=n_results# how many results to return
    )

    return results["documents"][0]

def run_tools_agent(system_prompt: Prompt):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_delivery_date",
                "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The customer's order ID.",
                        },
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    messages = []
    messages.append({"role": "system", "content": system_prompt.render()})
    model="gpt-4o"
    while True:
        human_input = input()
        messages.append({"role": "user", "content": human_input})
        response = openai.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        )
        messages.append({"role": "assistant", 'content': response.choices[0].message.content})
        print("agent:",response.choices[0].message.content)

def main(args):
    if args.create and args.run_agent:
       assert "Only one should be specify. Either create the Embeddings or run the Agent." 






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="This is a description of the program.")
    subparsers = parser.add_subparsers(title= "Commands", dest="command")
    create_parser = subparsers.add_parser("create", help="Create the Collection")
    create_parser.add_argument('--database_dir')
    create_parser.add_argument('--collection_name')
    create_parser.add_argument('--pdf_file')
    agent_parser = subparsers.add_parser('agent', help='Run the AI Agent')
    agent_parser.add_argument('--num_responses')
    agent_parser.add_argument('')
    args = parser.parse_args()
    main(args)


