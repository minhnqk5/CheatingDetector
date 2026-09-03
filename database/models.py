from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey
)

from sqlalchemy.orm import declarative_base


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