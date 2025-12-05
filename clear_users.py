from database import SessionLocal
from models import User


def main() -> None:
    """
    Delete all rows from the `users` table.

    This is a one-off maintenance script intended for local/dev use to
    clear all registered users from the database.
    """
    db = SessionLocal()
    try:
        deleted = db.query(User).delete()
        db.commit()
        print(f"Deleted {deleted} users from the database.")
    finally:
        db.close()


if __name__ == "__main__":
    main()


