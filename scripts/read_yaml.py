import yaml
import os

def read_yaml_to_dict(filepath: str) -> dict:
    """Reads a YAML file and returns its contents as a dictionary."""
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data

def find_single_yaml_file(directory: str) -> str:
    """Finds the only YAML file in the given directory and returns its name."""
    yaml_files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.yaml', '.yml'))
    ]

    return yaml_files[0]

def read_yaml_in_experiment(dir: str):
    filename = find_single_yaml_file(dir)
    return read_yaml_to_dict(os.path.join(dir, filename))

