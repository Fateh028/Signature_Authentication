from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# SQLite database location
DATABASE_URL = f"sqlite:///{BASE_DIR / 'signature_auth.db'}"

# Password hashing configuration (bcrypt)
BCRYPT_ROUNDS = 12

# Similarity threshold for ORB-based signature matching (0..1).
# Higher value => stricter verification (fewer false accepts, more false rejects).
ORB_SIM_THRESHOLD = 0.6

# Preprocessing target size for signatures
SIG_WIDTH = 256
SIG_HEIGHT = 128


