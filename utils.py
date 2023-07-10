from vertexai.preview.language_models import InputOutputTextPair
import json


def update_context(input, context):
    return context + ';' + input + ';'
