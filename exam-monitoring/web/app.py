import os
from flask import Flask
from database.database import db


def create_app():

    app = Flask(__name__)

    # Secret key cho Flask session / flash message
    app.config["SECRET_KEY"] = "exam-monitoring-secret-key-change-this"

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    database_path = os.path.join(
        project_root,
        "data",
        "exam_monitoring.db"
    )

    app.config[
        "SQLALCHEMY_DATABASE_URI"
    ] = (
        "sqlite:///"
        + database_path
    )

    app.config[
        "SQLALCHEMY_TRACK_MODIFICATIONS"
    ] = False

    upload_folder = os.path.join(
        project_root,
        "data",
        "face_images"
    )

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    app.config[
        "FACE_UPLOAD_FOLDER"
    ] = upload_folder

    db.init_app(app)

    with app.app_context():
        from database.models import User
        db.create_all()

    from web.routes import register_routes

    register_routes(app)

    return app