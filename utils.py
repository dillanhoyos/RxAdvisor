from vertexai.preview.language_models import InputOutputTextPair
import json

n = 0

def get_examples(examples_file):
    global n
    examples = []
    example_dict = dict()
    with open(examples_file, 'r') as exampleMessages:
        example_dict = json.load(exampleMessages)

    for i in range(n, n+10):
        input_text = example_dict[i]['input_text']
        output_text = example_dict[i]['output_text']
        examples.append(
            InputOutputTextPair(
                input_text=input_text,
                output_text=output_text
            )
        )

    n = n + 10

    return examples
