import re
from flask import Flask, render_template, request, redirect, url_for, make_response
from werkzeug.utils import secure_filename
import fasttext
from nltk.tokenize import sent_tokenize
from own_code import *
import csv
import io
from flask.helpers import flash

app = Flask(__name__)
model = fasttext.load_model("/home/stevenwijaya/skripsiWebApps/skripsiApps/cc.id.300.bin")
app.config['ALLOWED_EXT'] = ["CSV"]
app.config['TESTING'] = True

@app.route("/")
def hello():
    return render_template("SingleArticle.html")

@app.route("/multi")
def multi():
    return render_template("MultipleArticle.html")

@app.route("/summarize" , methods = ['POST'])
def summarizer():
    compression_type = request.form['compression-form-type'] 
    ratio =  int(request.form['compression-form-value'])
    text  =  request.form['form-news']
    mmr_lambda = float(request.form['form-lambda'])

    if(compression_type != "fixed"):
        ratio = int((ratio) * len(sent_tokenize(text)) / 100) #for apps purpose
        if (ratio >= len(sent_tokenize(text))):
          ratio = len(sent_tokenize(text))
        if (ratio == 0):
          ratio = 1
    else:
        if(ratio > len(sent_tokenize(text))):
            return render_template("SingleArticle.html", summary = "Target summary Length is greater than you news length" , original_article = text)
    summary =  summarize(text , ratio , mmr_lambda, model)
    return render_template("SingleArticle.html", summary = summary , original_article = text)


@app.route('/summarize-bulk', methods=['POST'])
def upload():
    if request.files:
        file = request.files['form-file']
        compression_type = request.form['compression-form-type'] 
        ratio =  int(request.form['compression-form-value'])
        mmr_lambda = float(request.form['form-lambda'])


        if not allowed_file(file.filename):
            return render_template("MultipleArticle.html", summary = "This is not csv file")

        else: 
            filename = secure_filename(file.filename)
            filename, ext = filename.split(".")
            data_set = file.read().decode('UTF-8')
            io_string = io.StringIO(data_set)
            teks_list = []
            for row in csv.reader(io_string):
                teks_list.append(row[0])
            summary = bulk_summarize(teks_list, compression_type, ratio, mmr_lambda, model)
            output = make_response(summary.to_csv(index=False))
            output.headers["Content-Disposition"] = "attachment; filename="+filename+".csv"
            output.headers["Content-type"] = "text/csv"
            return output

    return render_template('MultipleArticle.html')

def allowed_file(filename):
    if not "." in filename:
        return False
    
    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config['ALLOWED_EXT']:
        return True
    else:
        return False

if __name__ == '__main__':
    app.run(debug=True)
