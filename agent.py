from openai import OpenAI
from typing import Dict, List, Union, Tuple
from github_client import GithubClient
from typing import Dict, Optional
import datetime
from bs4 import BeautifulSoup
import requests
import logging
class UndefinedVariableError(Exception):
    pass


class Prompt:
    def __init__(self, instructions: str, variables: Optional[Dict[str, Union[str, 'Prompt']]] = None) -> None:
        self.instructions = instructions
        self.variables = variables

    def render(self) -> str:
        try:
            if self.variables is None:
                return self.instructions
            else:
                # Prepare a dictionary where we will store the rendered variables
                rendered_variables = {}

                for key, value in self.variables.items():
                    if isinstance(value, Prompt):
                        # If the value is a Prompt, render it
                        rendered_variables[key] = value.render()
                    else:
                        # Otherwise, just use the string value
                        rendered_variables[key] = value

                # Format the instructions with the rendered variables
                return self.instructions.format(**rendered_variables)
        except KeyError as e:
            # Raise a custom error if a variable is not defined
            raise UndefinedVariableError(f"Variable '{e.args[0]}' is not defined") from e

    def __add__(self, other: 'Prompt') -> 'Prompt':
        # Combine the instructions
        combined_instructions = self.instructions + " " + other.instructions
        
        # Combine the variables
        combined_variables = {**(self.variables or {}), **(other.variables or {})}
        
        # Return a new Prompt instance with the combined instructions and variables
        return Prompt(combined_instructions, combined_variables)


        

def openai_chat_completion(prompt: Prompt) -> str:
    client = OpenAI()
    prompt_text = prompt.render()
    response = client.chat.completions.create(
    model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response.choices[0].message.content


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
    First it should have a section called Work accomplish. In here it should reference the things implemented in the github repository
    as well as the concepts learned or applied. Ideally each commit would be a bullet point to consider but that can depend.

    After that it will have another section called Things Learned:
     In here  we will explain in more detail and in paragah form the things doned during the week. This part should have 
     mentioned the references mentioned on the data. It will reference it in IEEE form.

     Finally a section called Comclusions and Comments. In this part you will have a very reduced sumarry of all the things 
     that where work during the week and you will also add the Meta Notes that the person put that week.

     Layout and Structure:
        # PandaHat Adversarial Notebook
        #### Name: (persons_name)
        #### date: (start_date) - (end_date)
        ### Relevant Topics: (relevant_topics)
    
        (Agent_Generations)
    
        All that data will be below:
"""
    return  Prompt(instructions)
    

def github_data_prompt(gh_token: str, repo_url: str, branch: str, commits: Optional[int], date_start: Optional[str], date_end: Optional[str]) -> Prompt:
    instructions = """
    The following data is taken from the GitHub RESTful API. It includes information from a specific repository in a structured format:
    Format of the Data:
    Repository URL: {repo_url}
    Branch: {branch}
    Number of Commits: {num_commits}

    Data:
    {GH_DATA}
    """
    
    # Initialize the GitHub client
    client = GithubClient(token=gh_token)
    
    # Fetch commits with diffs, filtering by date if commits is not specified
    github_retrieved_data = client.get_commits_and_diffs(
        gh_repo_url=repo_url,
        branch=branch,
        num_commits=commits,
        date_start=date_start,
        date_end=date_end
    )
    
    # Format the retrieved commit data for inclusion in the prompt
    formatted_data = "\n\n".join([str(commit) for commit in github_retrieved_data])

    variables = {
        'repo_url': repo_url,
        'branch': branch,
        'num_commits': commits if commits else "Filtered by date",
        'GH_DATA': formatted_data
    }
    
    return Prompt(instructions, variables)


def references_prompt(links: list) -> Prompt:
    instructions = """
    The following data is extracted from the provided reference links. Each link contains important information that will be used to
    explain the technical concepts implemented in the code. The content extracted will be summarized and referenced in the final markdown document.

    Data extracted:
    {REFERENCES_DATA}
    """

    references_data = []
    
    for link in links:
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Example extraction logic:
            # Extract the title of the page
            title = soup.title.string if soup.title else "No Title"
            
            # Extract all paragraph texts
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])

            # Combine title and content
            references_data.append(f"Title: {title}\nContent: {content[:500]}...")  # Limiting content length for brevity

        except Exception as e:
            references_data.append(f"Error retrieving data from {link}: {e}")

    variables = {
        'REFERENCES_DATA': "\n\n".join(references_data)
    }

    return Prompt(instructions, variables)



def metadata_notes_prompt(files_path: str) -> Prompt:
    instructions = """
    This is a text file that has information about the notebook in general. It will not have a specific structure
    but it should be considered for the conclusions and others.
    Data of the file in here:
    {text_data}
    """
    with open(files_path, "r") as f:
        data = f.read()
    return Prompt(instructions, {'text_data': data})
    

    



def response_checker(response: str, instruction_prompt: Prompt )-> str:
    instructions = """
    You are given a response I would want you to revised if the initial instructions 
    from the prompt were meet. If that the case could you list what where the things that dindt
    work?
    
    INSTRUCTIONS Prompt used:
    {instruction_prompt}
    """
    checker_prompt = Prompt(instructions, {'instruction_prompt': instruction_prompt})
    return openai_chat_completion(checker_prompt)
def improve_response_with_checker(checker_response: str, genenerated_response: str)-> str:
    instructions = """
    I will give you two responses. the first one is a first generation of a text the second one is the response
    from a critique of the things that are wrong or could improve. With this information you will create a new response
    considering  (If necesarry) the critique maded, if it is something very subtle or not very important you will ignore it.
    Generated Response:
    {generated_response},
    Critique:
    {checker_response}
    """
    prompt = Prompt(instructions, {'generated_response': genenerated_response, 'checker_response': checker_response})
    return openai_chat_completion(prompt)





def notebook_pipeline(checker_iterations: int, gh_token: str, repo_url: str, branch: str, commits: Optional[int], date_start: Optional[str], date_end: Optional[str], reference_links: List[str], metadata_file_path: str) -> List[Tuple[str, str, str]]:
    # Generate the initial prompt components
    initial_prompt = general_instruction_prompt()
    github_data = github_data_prompt(gh_token, repo_url, branch, commits, date_start, date_end)
    references = references_prompt(reference_links)
    metadata = metadata_notes_prompt(metadata_file_path)
    final_prompt = initial_prompt + github_data + references + metadata
    responses = []
    response = openai_chat_completion(final_prompt)
    
    for i in range(checker_iterations):
        checker_response = response_checker(response, initial_prompt)
        revised_response = improve_response_with_checker(checker_response, response)
        responses.append((response, checker_response, revised_response))
        response = revised_response
    
    return responses
