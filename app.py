from flask import Flask, render_template, request, jsonify

import google.cloud.aiplatform as aiplatform
from google.auth import credentials
from google.oauth2 import service_account
import openai
import vertexai
from vertexai.preview.language_models import ChatModel

from utils import get_examples

import json

app = Flask(__name__)

examples_file = '/home/karthi/work/vertexAI/RxAdvisor/message.json'
examples = []

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    # return get_Chat_response(input)
    return get_Vertex_response(input)

prompt = "here is a prompt"

conversation = [
    {'role': 'user', 'content': prompt},
]

def get_Vertex_response(case):
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    examples = get_examples(examples_file)

    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat(
    context="""you provide medical remedies and medicine names to users based on their symptoms.""",
    examples=examples
    )
    response = chat.send_message(case, **parameters)
    chat.message_history.append(case)
    return response.text

def get_Chat_response(case):
    message_object = {'role': 'user', 'content': case}
    conversation.append(message_object)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=100,
        n=1,
        stop=None
    )
    assistant_reply = response.choices[0].message["content"]
    conversation.append({'role': 'assistant', 'content': assistant_reply})
    return jsonify(str(response.choices[0].message["content"].strip()))

if __name__ == '__main__':
    
    # Load the service account json file
    # Update the values in the json file with your own
    with open(
        "service-account-key.json"
    ) as f:  # replace 'serviceAccount.json' with the path to your file if necessary
        service_account_info = json.load(f)

    my_credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    # Initialize Google AI Platform with project details and credentials
    aiplatform.init(
        credentials=my_credentials,
    )
    vertexai.init(project="vextex-ai", location="us-central1")

    app.run()