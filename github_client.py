import requests
from typing import Dict, List, Tuple
import re
from datetime import datetime

class GithubCommitResponse:
    def __init__(self, message: str, committed_date: str, sha: str, author_name: str, author_email: str, url: str, diffs: List[Dict[str, str]]):
        self.message = message
        self.committed_date = committed_date
        self.sha = sha
        self.author_name = author_name
        self.author_email = author_email
        self.url = url
        self.diffs = diffs

    @staticmethod
    def from_dict(response_dict: dict, diffs: Dict[str, List[Dict[str, str]]]) -> List['GithubCommitResponse']:
        commits = []
        for commit_data in response_dict:
            sha = commit_data['sha']
            commit = GithubCommitResponse(
                message=commit_data['commit']['message'],
                committed_date=commit_data['commit']['committer']['date'],
                sha=sha,
                author_name=commit_data['commit']['author']['name'],
                author_email=commit_data['commit']['author']['email'],
                url=commit_data['html_url'],
                diffs=diffs.get(sha, [])
            )
            commits.append(commit)
        return commits

    def __repr__(self):
        diffs_str = "\n\n".join(
            [f"File: {diff['filename']} - Additions: {diff['additions']}, Deletions: {diff['deletions']}\nPatch:\n{diff['patch']}"
             for diff in self.diffs]
        )
        return (f"Commit({self.sha[:7]}): {self.message}\n"
                f"Author: {self.author_name} <{self.author_email}>\n"
                f"Date: {self.committed_date}\n"
                f"URL: {self.url}\n"
                f"Diffs:\n{diffs_str}\n")


class GithubClient:
    def __init__(self, token: str) -> None:
        self.base_url = "https://api.github.com"
        self.token = token

    def get_commits_and_diffs(self, gh_repo_url: str, branch: str, num_commits: int = None, date_start: str = None, date_end: str = None) -> List[GithubCommitResponse]:
        # Extract owner and repository name using regex
        owner, repo_name = self.extract_owner_repo(gh_repo_url)

        # Construct the REST API URL for listing commits
        url = f"{self.base_url}/repos/{owner}/{repo_name}/commits"
        params = {
            "sha": branch,
            "per_page": num_commits if num_commits else 100,  # Max per page is 100
        }

        # Execute the request to get commits
        response = self.run_rest_request(url, params)

        # Filter by date range if num_commits is not specified
        if not num_commits and date_start and date_end:
            date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
            date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
            response = [commit for commit in response if date_start_dt <= datetime.strptime(commit['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ") <= date_end_dt]

        # Fetch diffs for each commit
        diffs = {}
        for commit_data in response:
            sha = commit_data['sha']
            diffs[sha] = self.get_commit_diff(gh_repo_url, sha)

        # Convert the dictionary to a list of GithubCommitResponse objects with diffs
        commits_with_diffs = GithubCommitResponse.from_dict(response, diffs)
        return commits_with_diffs

    def get_commit_diff(self, gh_repo_url: str, commit_sha: str) -> List[Dict[str, str]]:
        # Extract owner and repository name using regex
        owner, repo_name = self.extract_owner_repo(gh_repo_url)

        # Construct the REST API URL for retrieving the commit diff
        url = f"{self.base_url}/repos/{owner}/{repo_name}/commits/{commit_sha}"

        # Execute the request
        response = self.run_rest_request(url)

        # Extract diff information
        files = response.get("files", [])
        diffs = [
            {
                "filename": file["filename"],
                "additions": file["additions"],
                "deletions": file["deletions"],
                "patch": file.get("patch", "No patch available")
            }
            for file in files
        ]
        return diffs

    def extract_owner_repo(self, gh_repo_url: str) -> Tuple[str, str]:
        """
        Extracts the owner and repository name from a GitHub URL.
        Example: https://github.com/octocat/Hello-World
        Returns: owner (octocat), repo_name (Hello-World)
        """
        pattern = r"https://github\.com/([^/]+)/([^/]+)"
        match = re.match(pattern, gh_repo_url)
        if match:
            owner = match.group(1)
            repo_name = match.group(2)
            return owner, repo_name
        else:
            raise ValueError("Invalid GitHub repository URL")

    def run_rest_request(self, url: str, params: Dict[str, str] = None) -> Dict:
        # Set up the request headers
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Make the request to the GitHub REST API
        response = requests.get(url, headers=headers, params=params)

        # Check for errors
        if response.status_code != 200:
            raise Exception(
                f"Request failed, status code {response.status_code}\n{response.text}"
            )

        # Return the JSON response
        return response.json()


if __name__ == "__main__":
    import os
    gh = os.environ['GH_KEY']
    client = GithubClient(token=gh)

    # Example of getting commits with diffs
    commits_with_diffs = client.get_commits_and_diffs(
        "https://github.com/Cruiz102/chat-api", 
        "main", 
        num_commits=2
    )
    for commit in commits_with_diffs:
        print(commit)
        print("\n")
