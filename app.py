import json
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask import Flask, Response, request
from model import BSM_Model
from variables import*

app = Flask(__name__)

try:
    client = MongoClient(db_url)
    db = client[database]
    client.server_info()

except:
    print("Error: Unable to connect to database.")

model = BSM_Model()
model.run()

@app.route("/books", methods=["POST"])
def predictions():
    try:
        book_data = request.get_json()
        response = model.predict_book(book_data)
        return Response(
                    response=response, 
                    status=500, 
                    mimetype="application/json"
                    )

    except Exception as e:
        print(e)

if __name__ == '__main__':
    app.run(debug=True, host=host)