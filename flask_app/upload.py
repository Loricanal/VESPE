from flask import Flask, render_template, request
from werkzeug import secure_filename
from flask import request, jsonify
import os
import subprocess
from subprocess import PIPE

datasets_folder = "../datasets"

email = ""


folders = list()
for x in os.walk(datasets_folder):
    f = x[0]
    f = f.split('/')[-1]
    try:
        folders.append(int(f))
    except:
        pass



if len(folders):
    max_f = str(max(folders)+1)
else:
    max_f = str(0)


path = "/".join([datasets_folder,max_f]) + "/"


os.makedirs(path)


app = Flask(__name__)

@app.route("/")
def get_index():
   return render_template('upload.html')
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global path
    global email
    if request.method == 'POST':
        f_data = request.files['data']
        #fname_data = secure_filename(f.filename)
        f_data.save(path+"data.csv")
        f_target = request.files['target']
        f_target.save(path+"target.csv")
        f_specification = request.files['specification']
        f_specification.save(path+"specifications.json")
        email = request.form['email']
        return render_template('success.html')


@app.route('/success', methods = ['GET', 'POST'])
def rendered_success():
    global max_f,email
    p = subprocess.Popen("cd ..; "+
        "python3 compute_Models_Scores.py "+max_f+" > output.txt; " +
        "python3 compute_ShapModels.py "+max_f+" > output.txt; " +
        "python3 send_Email.py " + email + " 1",shell=True,stdout=PIPE,stderr=PIPE)



    (out,err) = p.communicate()
    if p.returncode != 0:
        p = subprocess.Popen(["python3 send_Email.py " + email + " 0"])
    exit()


        
if __name__ == '__main__':
   app.run(debug = True)