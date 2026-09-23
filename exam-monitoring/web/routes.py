import os
import uuid
import cv2
import numpy as np
from flask import (render_template, request, redirect, url_for, flash, session, jsonify)
from werkzeug.security import (generate_password_hash, check_password_hash)
from database.database import db
from database.models import User
from face.face_embedding import (FaceEmbeddingService)
from monitoring.monitoring_service import (MonitoringService)


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

monitoring_sessions = {}



def allowed_file(filename):
    if "." not in filename:
        return False
    extension = (filename.rsplit(".", 1)[1].lower())
    return extension in ALLOWED_EXTENSIONS


def login_required():
    return "user_id" in session


def register_routes(app):
    face_service = (FaceEmbeddingService())


    @app.route("/")
    def index():
        if "user_id" in session:
            return redirect(url_for("home"))
        return redirect(url_for("login"))

    

    @app.route("/login", methods=["GET", "POST"])
    def login():

        if request.method == "GET":
            return render_template("login.html")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        if user is None:
            flash("Username hoặc password không đúng.", "error")
            return redirect(url_for("login"))
        
        if not check_password_hash(user.password_hash, password):
            flash("Username hoặc password không đúng.", "error")
            return redirect(url_for("login"))
        

        session["user_id"] = user.id
        session["username"] = (user.username)
        return redirect(url_for("home"))

    

    @app.route("/register", methods=["GET", "POST"])
    def register():

        if request.method == "GET":
            return render_template("register.html")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        face_file = request.files.get("face_image")


        # ----------------------------------
        # VALIDATE
        # ----------------------------------
        if not username:
            flash("Vui lòng nhập username.", "error")
            return redirect(url_for("register"))
        if not password:
            flash("Vui lòng nhập password.", "error")
            return redirect(url_for("register"))
        if face_file is None:
            flash("Vui lòng upload ảnh khuôn mặt.", "error")
            return redirect(url_for("register"))
        if face_file.filename == "":
            flash("Vui lòng chọn ảnh.", "error")
            return redirect(url_for("register"))
        if not allowed_file(face_file.filename):
            flash("Chỉ chấp nhận JPG, JPEG hoặc PNG.", "error")
            return redirect(url_for("register"))

        
        # ----------------------------------
        # CHECK USER
        # ----------------------------------
        existing_user = (User.query.filter_by(username=username).first())
        if existing_user is not None:
            flash("Username đã tồn tại.", "error")
            return redirect(url_for("register"))

        
        # ----------------------------------
        # SAVE IMAGE
        # ----------------------------------
        extension = (face_file.filename.rsplit(".", 1)[1].lower())
        filename = (f"{uuid.uuid4().hex}" f".{extension}")
        upload_folder = app.config["FACE_UPLOAD_FOLDER"]
        image_path = os.path.join(upload_folder, filename)
        face_file.save(image_path)


        # ----------------------------------
        # GENERATE EMBEDDING
        # ----------------------------------
        try:
            embedding = (face_service.generate_embedding(image_path))
        except ValueError as error:
            if os.path.exists(image_path):
                os.remove(image_path)
            flash(str(error), "error")
            return redirect(url_for("register"))
        except Exception as error:
            if os.path.exists(image_path):
                os.remove(image_path)
            print("Face embedding error:", error)
            flash("Có lỗi khi xử lý khuôn mặt.", "error")
            return redirect(url_for("register"))

        
        # ----------------------------------
        # CREATE USER
        # ----------------------------------
        embedding_bytes = (face_service.embedding_to_bytes(embedding))
        user = User(
            username=username,
            password_hash=(generate_password_hash(password)),
            face_image_path=image_path,
            face_embedding=(embedding_bytes)
        )
        db.session.add(user)
        db.session.commit()


        # ----------------------------------
        # AUTO LOGIN
        # ----------------------------------
        session["user_id"] = user.id
        session["username"] = (user.username)
        return redirect(url_for("home"))



    

    @app.route("/home")
    def home():
        if not login_required():
            return redirect(url_for("login"))
        user = User.query.get(session["user_id"])
        if user is None:
            session.clear()
            return redirect(url_for("login"))
        return render_template("home.html", username=user.username)



    @app.route("/api/monitor/start", methods=["POST"])
    def start_monitoring():
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Unauthorized"}), 401
        
        user_id = session["user_id"]
        user = User.query.get(user_id)

        if user is None:
            return jsonify({"success": False, "message": "User not found"}), 404
        if user.face_embedding is None:
            return jsonify({"success": False, "message": "User chưa có face embedding."}), 400


        
        # ----------------------------------
        # STOP OLD SESSION
        # ----------------------------------
        old_service = (monitoring_sessions.get(user_id))
        if old_service is not None:
            old_service.close()


        # ----------------------------------
        # LOAD EMBEDDING
        # ----------------------------------
        embedding = (face_service.bytes_to_embedding(user.face_embedding))
        if embedding is None:
            return jsonify({"success": False, "message": "Không thể đọc face embedding."}), 400
        embedding = np.asarray(embedding, dtype=np.float32)


        # ----------------------------------
        # CREATE MONITORING SERVICE
        # ----------------------------------
        service = MonitoringService(reference_embedding=embedding)
        monitoring_sessions[user_id] = service
        return jsonify({"success": True, "message": "Monitoring started."})



    

    @app.route("/api/monitor/frame", methods=["POST"])
    def monitor_frame():

        if "user_id" not in session:
            return jsonify({"success": False, "message": "Unauthorized"}), 401
        user_id = session["user_id"]
        service = (monitoring_sessions.get(user_id))
        if service is None:
            return jsonify({"success": False, "message": "Monitoring chưa được bật."}), 400


        
        # ==================================
        # READ IMAGE
        # ==================================
        if "frame" not in request.files:
            return jsonify({"success": False, "message": "Không có frame."}), 400
        file = request.files["frame"]
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"success": False, "message": "Frame rỗng."}), 400


        
        # ==================================
        # DECODE IMAGE
        # ==================================
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "message": "Không thể decode frame."}), 400


        
        # ==================================
        # AI PROCESSING
        # ==================================
        try:
            result = (service.process_frame(frame))
        except Exception as error:
            print("Monitoring error:", error)
            return jsonify({"success": False, "message": "Lỗi xử lý AI."}), 500
        return jsonify({"success": True, "status": result["status"], "events": result["events"]})





    @app.route("/api/monitor/stop", methods=["POST"])
    def stop_monitoring():
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Unauthorized"}), 401
        user_id = session["user_id"]
        service = (monitoring_sessions.pop(user_id, None))
        if service is not None:
            service.close()
        return jsonify({"success": True, "message": "Monitoring stopped."})

    

    @app.route("/logout")
    def logout():
        user_id = session.get("user_id")
        # Stop monitoring first
        if user_id is not None:
            service = (monitoring_sessions.pop(user_id, None))
            if service is not None:
                service.close()
        session.clear()
        return redirect(url_for("login"))
