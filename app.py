from flask import Flask, render_template, request, jsonify
import openai



app = Flask(__name__)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

prompt = "here is a prompt"

conversation = [
    {'role': 'user', 'content': prompt},

]



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
    app.run()