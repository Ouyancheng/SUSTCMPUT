"""
A Flask server for generating QR code
"""
import argparse
import io
import uuid
import flask
# from werkzeug.utils import secure_filename
import ics
import pyqrcode
import verification

app = flask.Flask(__name__)
debug = False
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
# app.config["UPLOAD_FOLDER"] = "uploads"


@app.route("/", methods=["GET"])
@app.route("/hello/<name>", methods=["GET"])
def hello(name: str = None):
    """
    Root handler
    """
    return flask.render_template("hello.html", name=name)


@app.route("/getcode/<uuid:target>", methods=["GET"])
def getcode(target: uuid.UUID):
    """
    Generate a random UUID string and encode it into a QR code;
    The QR code is stored in a stream.
    """
    stream = io.BytesIO()
    uuid_str = str(target)
    uuid_qrcode = pyqrcode.create(uuid_str)
    uuid_qrcode.png(stream, scale=8)
    stream.seek(0)
    return flask.send_file(stream, mimetype='image/png')


@app.route("/verify/<uuid:target>", methods=["GET"])
def verify(target: uuid.UUID):
    """
    Verify a UUID string
    """
    # test_uuid: 1bbff1ac-0a0a-422e-81c6-06514d563094
    if verification.verify_uuid(str(target)):
        return "OK"
    return "FAILED"


@app.route("/uploadics", methods=["POST"])
def upload_ics():
    """
    Route point to handle ics file upload
    """
    if "ics" not in flask.request.files:
        return "upload_failed: no file is uploaded"
    file_obj = flask.request.files["ics"]

    # filename = secure_filename(file_obj.filename)
    # save_path = "{}/{}".format(app.config["UPLOAD_FOLDER"], filename)
    # file_obj.save(save_path)
    try:
        cal = ics.Calendar(file_obj.read().decode('iso-8859-1'))
    except Exception as e:
        print(e)
        return "process_failed: not a valid ics file"
    uuid_str = verification.process_ics(cal)
    if uuid_str is None:
        return "process_failed: internal error"
    # return flask.send_file(qrcode_stream, mimetype="image/png")
    return flask.url_for("getcode", target=uuid_str)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p", "--port", help="Specify the server port number you need to listen, default is 8080")
    arg_parser.add_argument(
        "-d", "--debug", action="store_true", help="Specify whether it's in debug mode, defaut is False")
    args = arg_parser.parse_args()
    if args.debug:
        debug = args.debug
        print("Server running in debug mode")
    app.debug = debug
    service_ip = None if debug else '0.0.0.0'
    service_port = 8080 if args.port is None else int(args.port)
    app.run(host=service_ip, port=service_port)
