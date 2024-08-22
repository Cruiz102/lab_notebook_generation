from openai import OpenAI
from typing import Dict
from github_client import GithubClient
from typing import Dict, Optional
import datetime
class UndefinedVariableError(Exception):
    pass

class Prompt:
    def __init__(self, instructions: str, variables: Optional[Dict[str, str]] = None) -> None:
        self.instructions = instructions
        self.variables = variables

    def render(self) -> str:
        try:
            if self.variables is None:
                return self.instructions
            else:
                # Attempt to format the instructions with the provided variables
                return self.instructions.format(**self.variables)
        except KeyError as e:
            # Raise a custom error if a variable is not defined
            raise UndefinedVariableError(f"Variable '{e.args[0]}' is not defined") from e

    def __add__(self, other: 'Prompt') -> 'Prompt':
        # Combine the instructions
        combined_instructions = self.instructions + " " + other.instructions
        
        # Combine the variables
        combined_variables = {**self.variables, **other.variables}
        
        # Return a new Prompt instance with the combined instructions and variables
        return Prompt(combined_instructions, combined_variables)

def chat_completion(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']


def general_instruction_prompt() -> Prompt:
    instructions = """
    This is a program for generating a weekly research notebook. The purpose is to help in the generation of our weekly research
    notebook. The Research group is an undergratudate machine learning team called PandaHat Adversarial. We would want that given 
    a given data you would use it to create a Markdown notebook with a detail sumaryy of the work done.

    You will receive a sample of the following data:

    - Github data: You will receive two types of data. Commits history and their Diff. This means that given a github
                    data of the commits done and their diff data for the files changes you will create a sumary of the work
                    done  in programming and you should explain the concepts and the ideas implemented into the code.

    -  References: Every Machine Learning Idea implemented into the github repository should be defined what is it and how it
                    was used in the code. For that the user will give you a list of important references that will help you detailing
                    the tecnhical stuffs.

    - Meta Notes: You will also receive  the notes of the person. In here you can expect details about what things where difficult to understand
                    , things that call the attention of the person and ideas he wanted to annotated. Considering this data is very important.

    Markdown Formatting Instructions:

    
"""


def github_data_prompt(gh_token: str, repo_url: str, branch: str, commits: Optional[int], date_start: str, date_end: str) -> Prompt:
    instructions = """
    The following data is taken from the GitHub RESTful API. It includes information from a specific repository in a structured format:
    Format of the Data:
    Repository URL: 
    Branch: 
    Number of Commits:
    

    Data:
    {GH_DATA}
    """
    
    # Convert date strings to datetime objects for comparison
    date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
    date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    
    # Initialize the GitHub client
    client = GithubClient(token=gh_token)
    
    # Fetch commits with diffs
    github_retrived_data = client.get_commits_and_diffs(repo_url, branch, commits)
    
    variables = {
        'GH_DATA': github_retrived_data
    }
    
    return Prompt(instructions, variables)


def create_topics(initial_prompt: Prompt) -> str:
    instructions = """
    {}
    Using the previous data. Please do the following task.
    Create me a list of the main topics discussed on the references and  a list of the 
    concepts implemented in the code.
    """


