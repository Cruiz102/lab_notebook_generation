from openai import OpenAI
from typing import Dict, List, Union, Tuple
from github_client import GithubClient
from typing import Dict, Optional
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