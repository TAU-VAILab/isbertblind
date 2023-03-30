import re
from pathlib import Path
from typing import Optional, Pattern, List
from operator import truth
from setuptools import setup

REQUIREMENTS_FILE_NAME: str = 'requirements.txt'
requirements_file_regex: Pattern = re.compile(r"""^
    (?P<prefix>-r)
    \s+
    (?P<file_name>[\.\w/-]+)
    $""", re.VERBOSE)


def get_dependencies_list(requirements_file_path: Optional[str] = None) -> List[str]:
    """
    This function iterate over the given requirements file
    and collects all listed requirements.
    In case the caller did not feed the function with a requirements file
    the function will use the default requirements.txt file
    which is expected to be located in the same folder as the setup.py file.
    """
    requirements_list: List[str] = []

    # set default file path
    if not requirements_file_path:
        requirements_file_path = Path(__file__).parent / REQUIREMENTS_FILE_NAME

    # read the given requirements file
    with open(requirements_file_path, 'r') as stream:
        required = stream.read().splitlines()

    # Parse each requirement individually
    # Supported use cases:
    #   * Each line should hold a single requirement
    #   * Ignore line that starts with '#'
    #   * Call the function recursively for nested
    #     requirements file -> line should starts with '-r'
    for requirement in filter(truth, required):
        requirement = requirement.strip()
        if requirement.startswith("#"):
            continue

        nested_file = requirements_file_regex.match(requirement)
        if nested_file is not None:
            requirements_list.extend(get_dependencies_list(nested_file.group('file_name')))
        else:
            requirements_list.append(requirement)

    return requirements_list


if __name__ == '__main__':
    setup(
        install_requires=get_dependencies_list()
    )
