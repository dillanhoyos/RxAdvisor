from flask import Flask, render_template, request, jsonify
import openai
import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair


app = Flask(__name__)


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
    vertexai.init(project="vextex-ai", location="us-central1")
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat(
    context="""you provide medical remedies and medicine names to users based on their symptoms.""",
    examples=[
        InputOutputTextPair(
                input_text="""A big trigger for the onset of acne is puberty. Better nutrition and living standards have seen the age of puberty, especially in girls, decrease significantly over the past 40 years. It is now not uncommon for girls as young as 7 to develop acne. Acne is also affecting more adults later in life and doctors are not sure why. A growing number of women have acne in their 30s, 40s, 50s, and beyond. What Causes Acne? Our body constantly makes and sheds skin. Normally, dead skin cells rise to the surface of the pore and just flake off our body. At puberty, hormones trigger the production of sebum - an oily substance that helps moisturize our skin. Sebum sticks dead skin cells together, increasing their chances of becoming trapped inside a pore. Clogged pores become blackheads, whiteheads or pimples. If bacteria are also present, redness and swelling can occur resulting in the progression of the pimple into a cyst or nodule. Who is More at Risk of Acne? Unfortunately, some people suffer from acne worse than others. Bad acne tends to run in families - your mother, father, aunt or uncle probably had severe breakouts when they were a teenager. Some people also have naturally higher hormone levels and make more sebum, so their skin pores are always clogging up. If you live in an area that gets very humid or have a job which exposes you to moist heat (such as in a food kitchen) or grease or tar (a mechanic or road worker) then you are more likely to get acne. Chin straps, headbands, and even hair products applied too close to the skin can precipitate a break out as several different medicines - most notably prednisone, phenytoin, and certain hormonal contraceptives that are high in androgens (for example, Microgestin 1.5/30 and the Depo-Provera shot). What are the Symptoms of Acne? Acne may appear on the face, forehead, chest, upper back or shoulders. The symptoms and severity of acne vary from person to person but may include: Whiteheads Blackheads Papules (small, red, tender bumps) Pimples (papules with pus at their tips) Nodules (large solid painful lumps beneath the skin surface Cystic lesions (painful pus-filled lumps beneath the skin’s surface). How is Acne Diagnosed? If your acne makes you shy or embarrassed, if you have a lot of acne, cysts or nodules on your face or back, or if over the counter products do not seem to work, see your doctor or a dermatologist as soon as you can. They can prescribe stronger topical or oral treatments that are much more effective than products you can buy at a drug store. Your doctor will look at your skin and ask about the history of your acne. It is a myth that you have to let acne run its course. Treatment helps prevent dark spots and permanent scars from forming as the acne clears. How is Acne Treated? The most important thing you can do to reduce the chance of breakouts is to take good care of your skin. This doesn\\\'t mean scrubbing it raw several times a day with soap. It means gently cleansing it with a mild soap-free wash twice a day, every day. If you play a lot of sport or work in a greasy or humid environment, cleanse your skin as soon as you finish training or right after work. Be gentle. You aim to cleanse away excess sebum and dead skin cells so they don\\\'t clog up your pores - not to irritate your skin even further.\"\"\",""",
                output_text="""The disease in the input text is acne. Acne is a common skin condition that occurs when your hair follicles become clogged with oil and dead skin cells. This can cause whiteheads, blackheads, pimples, and cysts. Acne is most common in teenagers, but it can also affect adults. There are many different treatments for acne, including over-the-counter medications, prescription medications, and lifestyle changes."""
            )
        ]
    )
    response = chat.send_message(case, **parameters)
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
    app.run()