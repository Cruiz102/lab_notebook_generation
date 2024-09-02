import openai
import chromadb
from tqdm import tqdm
from agent import Prompt
from PyPDF2 import PdfReader
import json

def create_collection(collection_name: str, database_path: str, pdf_file: str):
    client = chromadb.PersistentClient(path=database_path)
    collection = client.create_collection(collection_name)
    reader = PdfReader(pdf_file)
    num_pages = len(reader.pages)
    for i in tqdm(range(num_pages)):
        collection.add(
            documents=[reader.pages[i].extract_text()], 
            metadatas=[{"source": "laamps_documentation", 'page': i}],  # filter on these!
            ids=[f"laamps_page:${i}"],  # unique for each doc
        )

def chroma_query(collection_name: str, database_path: str, query: str, n_results: int):
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_collection(collection_name)
    results = collection.query(
        query_texts=[query],  # Chroma will embed this for you
        n_results=n_results  # how many results to return
    )
    return results

def additional_processing(function_result: dict):
    # Initialize an empty string to accumulate the formatted results
    processed_result = ""
    
    # Iterate over the list of documents
    for i in range(len(function_result['documents'][0])):
        # Extract page number from metadata if available
        page_info = function_result['metadatas'][0][i].get("page", "Unknown page")  # Assume metadata contains 'page' key
        text = function_result['documents'][0][i]  # Assume the document contains 'text' key
        
        # Format each document into a string
        processed_result += f"Result {i+1}, Page {page_info}:\n{text}\n\n"
    
    # Return the accumulated string
    return processed_result

def execute_function(function_name: str, parameters: dict, hard_coded_params: dict):
    if function_name == "chroma_query":
        collection_name = hard_coded_params['collection_name']  # Hardcoded collection name
        database_path = hard_coded_params['database_path']  # Hardcoded database path
        query = parameters.get("query")
        n_results = parameters.get("n_results")
        return chroma_query(collection_name, database_path, query, n_results)
    return "Function not recognized."


def run_tools_agent(system_prompt: Prompt, hard_coded_params: dict):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "chroma_query",
                "description": "Query a Chroma collection to retrieve documents based on a query string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query string to search the collection."
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "The number of results to return."
                        }
                    },
                    "required": ["query", "n_results"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    messages = []
    messages.append({"role": "system", "content": system_prompt.render()})
    model = "gpt-4o-mini"
    
    while True:
        human_input = input("You: ")
        messages.append({"role": "user", "content": human_input})
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        
        assistant_message = response.choices[0].message.content
        print('\n', response, '\n')
        
        # Check if the response includes a function call
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                parameters = json.loads(tool_call.function.arguments)
                
                # Execute the function with hard-coded parameters
                function_result = execute_function(function_name, parameters, hard_coded_params)
                
                # Additional processing on the function result
                processed_result = additional_processing(function_result)
                
                # Use the processed result to make another inference (calling the API again)
                messages.append({"role": "function", "name": function_name, "content": processed_result})
                inference_response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                
                inference_message = inference_response.choices[0].message.content
                messages.append({"role": "assistant", "content": inference_message})
                print("Agent:", inference_message)
        else:
            messages.append({"role": "assistant", 'content': assistant_message})
            print("Agent:", assistant_message)
        
        # Save conversation to a file
        with open('saved_conversation.txt', 'a') as f:
            f.write(f"User: {human_input}\n")
            f.write(f"Agent: {assistant_message}\n")
            if 'processed_result' in locals():
                f.write(f"Processed Result: {processed_result}\n")
                f.write(f"Inference based on Processed Result: {inference_message}\n")

def main(args):
    hard_coded_params = {
        'collection_name': args.collection_name,
        'database_path': args.database_dir
    }

    if args.command == "create":
        create_collection(args.collection_name, args.database_dir, args.pdf_file)
    elif args.command == "agent":
        system_prompt = """
        You are an advanced AI assistant integrated with a specialized database. When given data from the database, your task is to analyze the information and use it to develop a thoughtful and accurate response to the user's question.

        For each query:
        1. Analyze the provided data, which may include text, metadata, and other relevant details.
        2. Use this analysis to generate a well-informed, clear, and concise answer.
        3. If the data is insufficient or unclear, infer the most reasonable conclusion based on the context.
        4. Ensure your response is directly relevant to the user's question, providing specific details and explanations from the data where applicable.

        Your goal is to provide accurate, context-aware answers that are directly derived from the data provided.

        """
        system_prompt = Prompt(instructions=system_prompt)  # You may need to customize this based on your `Prompt` class
        run_tools_agent(system_prompt, hard_coded_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="This is a description of the program.")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # Parser for the 'create' command
    create_parser = subparsers.add_parser("create", help="Create the Collection")
    create_parser.add_argument('--database_dir', required=True, help="Path to the database directory")
    create_parser.add_argument('--collection_name', required=True, help="Name of the collection to create")
    create_parser.add_argument('--pdf_file', required=True, help="Path to the PDF file to parse")

    # Parser for the 'agent' command
    agent_parser = subparsers.add_parser('agent', help='Run the AI Agent')
    agent_parser.add_argument('--num_responses', required=True, help='Number of responses the agent should generate')
    agent_parser.add_argument('--database_dir', required=True, help="Path to the database directory")
    agent_parser.add_argument('--collection_name', required=True, help="Name of the collection to use")

    args = parser.parse_args()
    main(args)
