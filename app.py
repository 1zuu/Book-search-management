from model import BSM_Model
from flask import Flask, Response, request
from variables import*

app = Flask(__name__)


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