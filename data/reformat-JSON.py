import json
import ast

def convert_json_format(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    # Remove the extra opening curly brace
    content = content.lstrip('{')

    # Split the content into key-value pairs
    pairs = content.split(', ')

    json_data = {}
    for pair in pairs:
        # Split each pair into key and value
        key, value = pair.split(': ', 1)

        # Remove the extra quotes and convert the key to a string
        key = ast.literal_eval(key)

        # Check if the value has an extra closing curly brace and remove it
        if value.endswith('}'):
            value = value[:-1]

        # Convert the value to an integer
        value = int(value)

        json_data[key] = value

    with open(output_file, 'w') as file:
        json.dump(json_data, file, indent=2)

# Usage example
input_file = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/vocab.json'
output_file = '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/AI/Project-AI-NLP/vocab2.json'
convert_json_format(input_file, output_file)