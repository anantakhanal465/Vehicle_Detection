from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema():
    # `Base.metadata.create_all` only creates missing tables, not missing
    # columns on tables that already exist — needed here since
    # detection_records predates the image_url column and already holds
    # real history rows we don't want to lose.
    inspector = inspect(engine)

    if "detection_records" not in inspector.get_table_names():
        return

    columns = {c["name"] for c in inspector.get_columns("detection_records")}

    if "image_url" not in columns:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE detection_records ADD COLUMN image_url VARCHAR"
                )
            )
