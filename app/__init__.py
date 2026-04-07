import os

from flask import Flask
from werkzeug.exceptions import RequestEntityTooLarge

from .db import init_db
from .routes import bp
from .worker import recover_incomplete_jobs, start_worker


def create_app() -> Flask:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app.config["SECRET_KEY"] = os.environ.get("APP_SECRET_KEY", "change-me-in-production")
    app.config["BASE_DIR"] = base_dir
    app.config["DATA_DIR"] = os.path.join(base_dir, "data")
    app.config["JOB_DIR"] = os.path.join(base_dir, "data", "jobs")
    app.config["ASSET_DIR"] = os.path.join(base_dir, "data", "assets")
    app.config["DATABASE"] = os.path.join(base_dir, "data", "app.db")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("APP_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
    init_db(app.config["DATABASE"])
    recover_incomplete_jobs(app)
    app.register_blueprint(bp)
    start_worker(app)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_exc):
        return (
            "上传文件过大。当前上限为 "
            f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB，"
            "请压缩后重试，或改用链接生成。",
            413,
        )

    return app
