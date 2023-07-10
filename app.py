from flask import Flask, render_template, request, jsonify

import google.cloud.aiplatform as aiplatform
from google.auth import credentials
from google.oauth2 import service_account
import openai
import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

from utils import update_context


import json

app = Flask(__name__)

examples_file = '/home/karthi/work/vertexAI/RxAdvisor/message.json'
examples = []
context = 'You are a highly skilled health consultant, equipped with extensive knowledge of remedies and medicine names. Your task is to assist users by providing them with appropriate medical recommendations based on their symptoms. As a responsible consultant, you maintain a comprehensive understanding of the various drugs available in the market, their applications, and potential side effects. With your expertise, you aim to offer reliable advice and promote the well-being of those seeking assistance. Remember to provide accurate and relevant information, considering the unique needs and conditions of each individual.'

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
    global context
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat(
    context= context,
    examples=[            
            InputOutputTextPair(
                input_text="I having excess body fat, shortness of breath, sweating more than usual and snoring",
                output_text="Disease: Obesity \n\n Diagnosis:The most common way to determine if a person is affected by overweight or obesity is to calculate BMI, which is an estimate of body fat that compares a persons weight to their height.\n\n Medicine: Lorcaserin, Bontril Slow Release, Bupropion \n\n Treatment: Common treatments for overweight and obesity include losing weight through healthy eating, being more physically active, and making other changes to your usual habits. Weight-management programs may help some people lose weight or keep from regaining lost weight."
            )
        ]
    )
    response = chat.send_message(case, **parameters)
    chat.message_history.append(case)
    context = update_context(case, context)
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