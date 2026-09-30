from flask import Flask

import config


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    for d in (config.UPLOAD_DIR, config.OUTPUT_DIR, config.EVENTS_DIR, config.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    from web.routes import bp
    app.register_blueprint(bp)
    return app
