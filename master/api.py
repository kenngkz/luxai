from flask import Flask, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import time

from master.api_func.req_para import get_para, eval_args
from master.api_func.job_complete import manage_completion
from master.api_func.job_assign import assign
from master.api_func.upload import upload, upload2
from base_utils import path_join
from constants import POOL_DIR, MASTER_DATABASE_DIR, N_SELECT, N_BENCHMARKS, DEFAULT_PARAM_TEMPLATE

app = Flask(__name__)
app.secret_key = "super secret key"

HOST = "0.0.0.0"
PORT = 5001

app.config["UPLOAD_FOLDER"] = MASTER_DATABASE_DIR

@app.route('/job/request', methods=["GET", "POST"])
def handle_job_request():
    '''
    Returns a job. Performs internal job management.
    '''
    return assign()

@app.route('/job/report', methods=["GET", "POST"])
def handle_job_report():
    '''
    Returns a OK message. Performs internal job management. Updates eval_results for eval. Handle file upload for train.
    (update jobs in progress, add new jobs (evalstage completed -> new train jobs))
    To know if a stage has been fully evaluated, check for length of eval_results.
    '''
    p_criteria = {"req":["completed_job", "results"], "opt":{"n_select":N_SELECT, "n_benchmarks":N_BENCHMARKS, "param_template":DEFAULT_PARAM_TEMPLATE}}
    args = get_para(request.args, p_criteria)
    args = eval_args(args, ["completed_job", "n_select", "n_benchmarks", "param_template"])
    if args["completed_job"]["type"] != "train":
        args = eval_args(args, ["results"])
    manage_completion(**args)

    return "OK"

@app.route('/get/<path:file_path>', methods=["GET"])
def download_file(file_path):
    '''
    Returns a file for the client to download.
    Used to send models (model.zip files) to client
    '''
    if not os.path.exists(path_join(app.config["UPLOAD_FOLDER"], file_path)):
        print(f"File {path_join(app.config['UPLOAD_FOLDER'], file_path)} not found in database.")
    return send_from_directory(app.config["UPLOAD_FOLDER"], file_path, mimetype="zip", as_attachment=True)

@app.route('/upload', methods=["GET", "POST"])
def upload_file():
    '''
    Receive files uploaded from a client.
    '''
    p_criteria = {'req':[], 'opt':{'path':'./'}}
    subfolder_path = get_para(request.args, p_criteria)['path']  # subfolder to save the uploaded file.
    if subfolder_path[-1] != '/':
        subfolder_path += '/'

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(path_join(app.config['UPLOAD_FOLDER'], subfolder_path+filename))  # TODO: check if subfolder is needed?
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/upload2', methods=["GET", "POST"])
def upload_file2():
    '''
    Receive files uploaded from a client (version2).
    '''
    p_criteria = {'req':[], 'opt':{'path':None}}
    save_path = get_para(request.args, p_criteria)['path']  # save path for uploaded file
    return upload(app, request, save_path)

@app.route('/test', methods=["GET", "POST"])
def test():
    return "OK"

@app.route('/wait', methods=["GET", "POST"])
def wait():
    p_criteria = {"req":[], "opt":{"time":5}}
    args = get_para(request.args, p_criteria)
    args = eval_args(args, ["time"])
    time.sleep(args["time"])

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=True)