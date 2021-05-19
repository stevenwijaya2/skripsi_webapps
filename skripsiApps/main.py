from flask import Flask , render_template , request
app = Flask(__name__)
app.config['TESTING'] = True

@app.route("/")
def hello():
    return render_template("home.html")

@app.route("/summarize" , methods = ['POST'])
def summarizer():
    return request.form['ratio']

if __name__ == '__main__':
    app.run(debug=True)
