from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey
)

from sqlalchemy.orm import declarative_base
from database.database import db

Base = declarative_base()


class ExamSession(Base):

    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True)

    student_id = Column(String)

    start_time = Column(DateTime)

    end_time = Column(DateTime)


class BehaviorEvent(Base):

    __tablename__ = "behavior_events"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        Integer,
        ForeignKey("exam_sessions.id")
    )

    behavior_type = Column(String)

    start_time = Column(DateTime)

    end_time = Column(DateTime)

    duration = Column(Float)

    confidence = Column(Float)

    evidence_path = Column(String)



class User(db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    # --------------------------------------------------
    # Ảnh khuôn mặt gốc
    # --------------------------------------------------

    face_image_path = db.Column(
        db.String(500),
        nullable=True
    )

    # --------------------------------------------------
    # InsightFace embedding
    #
    # --------------------------------------------------

    face_embedding = db.Column(
        db.LargeBinary,
        nullable=True
    )

    def __repr__(self):

        return (
            f"<User "
            f"id={self.id} "
            f"username={self.username}>"
        )

