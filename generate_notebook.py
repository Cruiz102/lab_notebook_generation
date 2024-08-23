import argparse
import os
import requests
from agent import notebook_pipeline
from bs4 import BeautifulSoup
from typing import List

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
    parser.add_argument("--date_start", type=str, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--date_end", type=str, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--reference_file_path", type=str, required=True, help="Path to the text file containing reference links")
    parser.add_argument("--metadata_file_path", type=str, required=True, help="Path to the metadata notes file")
    parser.add_argument("--checker_iterations", type=int, default=1, help="Number of checker iterations")
    parser.add_argument("--output_folder", type=str, default="notebook_iterations", help="Folder to save the generated notebooks")
    
    args = parser.parse_args()

    # Use the provided token or fall back to the environment variable
    gh_token = args.gh_token or os.getenv("GITHUB_TOKEN")
    if not gh_token:
        raise ValueError("GitHub token must be provided either via --gh_token or GITHUB_TOKEN environment variable.")
    
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
    for i, (initial_response, checker_response, revised_response) in enumerate(responses):
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
        
        print(f"Iteration {i+1} has been saved to {file_path}")

if __name__ == "__main__":
    main()
