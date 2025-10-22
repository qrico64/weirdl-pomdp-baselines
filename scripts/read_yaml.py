from ruamel.yaml import YAML
import os

def _convert_to_dict(data):
    """Recursively convert ruamel.yaml objects to standard Python dict."""
    if isinstance(data, dict):
        return {key: _convert_to_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_to_dict(item) for item in data]
    else:
        return data

def read_yaml_to_dict(filepath: str) -> dict:
    """Reads a YAML file and returns its contents as a dictionary.

    Uses ruamel.yaml to preserve string literals like 'yes', 'no', 'on', 'off'
    as strings instead of converting them to booleans.
    """
    yaml = YAML()
    with open(filepath, 'r') as file:
        data = yaml.load(file)
    return _convert_to_dict(data)

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

