import argparse
import os
import requests
from openai import OpenAI
from typing import Dict, List, Union, Tuple
from github_client import GithubClient
from typing import Dict, Optional
import datetime
from bs4 import BeautifulSoup
import requests
import logging
from agent import Prompt


        

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
    

# Function to get the start and end dates for the current week (Sunday to Saturday)
def get_current_week_dates() -> Tuple[str, str]:
    today = datetime.date.today()
    start_of_week = today - datetime.timedelta(days=today.weekday() + 1)  # Sunday of the current week
    end_of_week = start_of_week + datetime.timedelta(days=6)  # Saturday of the current week
    return start_of_week.strftime("%Y-%m-%d"), end_of_week.strftime("%Y-%m-%d")

# Modified github_data_prompt function to support getting commits for the current week
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
    
    # If no date range is provided, fetch the current week’s commits
    if not date_start or not date_end:
        date_start, date_end = get_current_week_dates()
    
    # Initialize the GitHub client
    client = GithubClient(token=gh_token)
    
    # Fetch commits with diffs, filtering by date if commits are not specified
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
    

    
def format_final_response(raw_response: str) -> str:
    intructions = """Given the generation of the  notebook report I want you to answer the following questions. 
                    Make sure is not a very big response. Mention all the topics without explaining them for too long.
                    Work accomplished:
                        Hours worked: 
                        Problems encountered: 
                        What I learned:
                        Resources needed, if any: 
                        Plans for next week:

                        Raw_response:
                        {RAW_RESPONSE}
    """

    prompt = Prompt(intructions, {"RAW_RESPONSE":raw_response})

    return openai_chat_completion(prompt)
 




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
        answer_weekly_questions = format_final_response(revised_response)
        responses.append((response, checker_response, revised_response, answer_weekly_questions))
        response = revised_response
    
    return responses



def load_reference_links(file_path: str) -> List[str]:
    """Load reference links from a text file where each line is a different link."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Reference links file {file_path} does not exist.")
    
    with open(file_path, 'r') as f:
        links = [line.strip() for line in f.readlines() if line.strip()]
    
    if not links:
        raise ValueError(f"No valid links found in the file {file_path}.")
    
    return links

def check_reference_links(reference_links):
    """Check if reference links return valid content."""
    for link in reference_links:
        try:
            response = requests.get(link)
            if response.status_code != 200:
                raise ValueError(f"Error fetching {link}: Status code {response.status_code}")
            soup = BeautifulSoup(response.content, 'html.parser')
            if not soup.get_text(strip=True):
                raise ValueError(f"No valid content found in {link}")
        except Exception as e:
            raise ValueError(f"Error processing {link}: {e}")

def check_metadata_file(metadata_file_path):
    """Check if the metadata file exists and is non-empty."""
    if not os.path.isfile(metadata_file_path):
        raise FileNotFoundError(f"Metadata file {metadata_file_path} does not exist.")
    if os.path.getsize(metadata_file_path) == 0:
        raise ValueError(f"Metadata file {metadata_file_path} is empty.")

def main():
    parser = argparse.ArgumentParser(description="Generate a weekly research notebook with multiple iterations.")
    
    parser.add_argument("--gh_token", type=str, help="GitHub API token (if not specified, will read from GITHUB_TOKEN environment variable)")
    parser.add_argument("--repo_url", type=str, required=True, help="GitHub repository URL")
    parser.add_argument("--branch", type=str, required=True, help="Branch name")
    parser.add_argument("--commits", type=int, help="Number of commits to fetch", default=None)
    parser.add_argument("--date_start", type=str, help="Start date in YYYY-MM-DD format", default=None)
    parser.add_argument("--date_end", type=str, help="End date in YYYY-MM-DD format", default=None)
    parser.add_argument("--reference_file_path", type=str, required=True, help="Path to the text file containing reference links")
    parser.add_argument("--metadata_file_path", type=str, required=True, help="Path to the metadata notes file")
    parser.add_argument("--checker_iterations", type=int, default=1, help="Number of checker iterations")
    parser.add_argument("--output_folder", type=str, default="notebook_iterations", help="Folder to save the generated notebooks")
    
    args = parser.parse_args()

    # Use the provided token or fall back to the environment variable
    gh_token = args.gh_token or os.getenv("GITHUB_TOKEN")
    if not gh_token:
        raise ValueError("GitHub token must be provided either via --gh_token or GITHUB_TOKEN environment variable.")
    
    # Check if either commits or a valid date range is provided

    if not args.commits and (not args.date_start or not args.date_end):
        print("No commits or date range provided, defaulting to the current week's commits.")
        args.date_start, args.date_end = get_current_week_dates()  # Automatically set current week
    
    # Load and validate reference links
    reference_links = load_reference_links(args.reference_file_path)
    check_reference_links(reference_links)
    
    # Validate metadata file
    check_metadata_file(args.metadata_file_path)
    
    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Generate the notebook content
    responses = notebook_pipeline(
        checker_iterations=args.checker_iterations,
        gh_token=gh_token,
        repo_url=args.repo_url,
        branch=args.branch,
        commits=args.commits,
        date_start=args.date_start,
        date_end=args.date_end,
        reference_links=reference_links,
        metadata_file_path=args.metadata_file_path
    )
    
    # Save each iteration response to a separate file
    for i, (initial_response, checker_response, revised_response, answer) in enumerate(responses):
        file_name = f"iteration_{i+1}.md"
        file_path = os.path.join(args.output_folder, file_name)
        
        with open(file_path, 'w') as f:
            f.write(f"# Iteration {i+1}\n\n")
            f.write("## Initial Response\n")
            f.write(initial_response + "\n\n")
            f.write("## Checker Response\n")
            f.write(checker_response + "\n\n")
            f.write("## Revised Response\n")
            f.write(revised_response + "\n\n")
            f.write("## Asnwer to Notebook\n")
            f.write(answer + "\n\n")

        
        print(f"Iteration {i+1} has been saved to {file_path}")

if __name__ == "__main__":
    main()
