from base_utils import path_join

from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename
import os

def upload(app, request, save_path=None):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            if not save_path:
                save_path = filename
            save_path = path_join(app.config["UPLOAD_FOLDER"], save_path)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            file.save(save_path)
            return f"Upload complete. File saved to {save_path}"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''