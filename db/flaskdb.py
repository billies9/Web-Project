from flask_sqlalchemy import SQLAlchemy

from db.models import metadata

db = SQLAlchemy(metadata=metadata)
