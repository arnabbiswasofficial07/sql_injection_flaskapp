from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    name = request.form.get("name")
    password = request.form.get("password")

    # Finding the length of name and password
    len_name = len(str(name))
    len_password = len(str(password))

    # Finding the number of punctuations
    punctuation_array = ["<", ">", "<=", ">=", "=", "==", "!=", "<<", ">>", "|", "&", "-", "+", "%", "^", "*"]
    cpunctuation_name = 0
    cpunctuation_password = 0
    for ch in name:
        if ch in punctuation_array:
            cpunctuation_name += 1
    for ch in password:
        if ch in punctuation_array:
            cpunctuation_password += 1

    # Finding the number of keywords
    keywords_array = ["select", "update", "insert", "create", "drop", "alter", "rename", "exec", "order", "group",
                      "sleep", "count", "where"]
    ckeywords_name = 0
    ckeywords_password = 0
    for ch in name:
        if ch in keywords_array:
            ckeywords_name += 1
    for ch in password:
        if ch in keywords_array:
            ckeywords_password += 1

    # Displaying the output
    prediction_name = model.predict([[len_name, cpunctuation_name, ckeywords_name]])
    prediction_password = model.predict([[len_password, cpunctuation_password, ckeywords_password]])
    s = "You have given correct name and password"
    if prediction_name == 1 or prediction_password == 1:
        s = "You are trying sql injection attack"

    return render_template('index.html', prediction_text=s)


if __name__ == "__main__":
    app.run(debug=True)
