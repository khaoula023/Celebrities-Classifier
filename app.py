from flask import Flask, request, jsonify
#import util

app = Flask(__name__)

@app.route('/')
def classification():
    return "Hello every one"


if __name__ == "__main__":
    app.run(host='localhost', port=9874, debug=True)