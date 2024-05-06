

from flask import Flask, request, jsonify
from vectorstore import get_text_chunks, get_vectorstore
from model import get_conversation_chain
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

def get_url_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    url = data['url']
    question = data['question']

    # get text from URL
    text = get_url_text(url)

    # get the text chunks
    text_chunks = get_text_chunks(text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    conversation = get_conversation_chain(vectorstore)

    # get response to the question
    response = conversation({'question': question})
    chat_history = response['chat_history']
    bot_response = chat_history[-1].content

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)

