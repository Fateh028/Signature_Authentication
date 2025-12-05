# Signature-Based Authentication System

A secure web application that uses signature recognition for user authentication. This system combines traditional password authentication with biometric signature verification using advanced computer vision techniques.

## Features

- **Dual Authentication**: Username/password + signature verification
- **Advanced Signature Matching**: Uses multiple algorithms including:
  - ORB (Oriented FAST and Rotated BRIEF) keypoint matching
  - Shape-based features (Hu moments, contour analysis)
  - Pixel distribution analysis
  - SSIM (Structural Similarity Index) comparison
- **User Registration**: Two-step registration process with signature capture
- **Secure Storage**: Passwords are hashed using bcrypt, signatures are stored as encoded templates
- **Web Interface**: Modern, responsive UI with signature pad integration
- **FastAPI Backend**: High-performance async API framework

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Computer Vision**: OpenCV, NumPy, SciPy, scikit-image
- **Frontend**: HTML, CSS, JavaScript (Signature Pad)
- **Security**: bcrypt for password hashing

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository** (or download the project):
   ```bash
   git clone <your-repository-url>
   cd signature_auth
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The application uses default settings defined in `config.py`. You can modify:

- `ORB_SIM_THRESHOLD`: Similarity threshold for signature matching (default: 0.6)
  - Higher values = stricter verification (fewer false accepts, more false rejects)
  - Lower values = more lenient verification
- `SIG_WIDTH` and `SIG_HEIGHT`: Target dimensions for signature preprocessing
- `BCRYPT_ROUNDS`: Password hashing rounds (default: 12)

## Usage

1. **Start the application**:
   ```bash
   uvicorn app:app --reload
   ```
   Or if using Python directly:
   ```bash
   python -m uvicorn app:app --reload
   ```

2. **Access the application**:
   Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

3. **Register a new user**:
   - Go to the registration page
   - Enter a username and password
   - Draw your signature on the signature pad
   - Complete the registration process

4. **Login**:
   - Enter your username and password
   - Draw your signature to verify your identity
   - The system will compare your signature with the stored template

## Project Structure

```
signature_auth/
├── app.py                 # Main FastAPI application
├── auth.py                # Authentication utilities (password hashing)
├── config.py              # Configuration settings
├── database.py            # Database setup and session management
├── models.py              # SQLAlchemy models
├── signature_utils.py     # Signature processing and comparison algorithms
├── clear_users.py         # Utility script to clear user database
├── requirements.txt       # Python dependencies
├── signature_auth.db      # SQLite database (created automatically)
├── data/
│   └── fake_profiles.json # Sample user profile data
├── static/
│   ├── css/
│   │   └── styles.css     # Application styles
│   └── js/
│       └── signature_pad.js # Signature pad library
└── templates/
    ├── base.html          # Base template
    ├── register.html      # Registration form
    ├── register_signature.html # Signature capture for registration
    ├── login.html         # Login form
    ├── login_signature.html # Signature verification for login
    └── user_info.html     # User profile page
```

## How It Works

1. **Registration Process**:
   - User provides username and password
   - Password is hashed using bcrypt
   - User draws signature on the signature pad
   - Signature is preprocessed and features are extracted
   - Features are encoded and stored as a JSON template

2. **Authentication Process**:
   - User provides username and password
   - Password is verified against stored hash
   - User draws signature for verification
   - New signature features are extracted
   - Features are compared with stored template using multiple algorithms
   - If similarity score exceeds threshold, authentication succeeds

3. **Signature Comparison**:
   - Multiple comparison methods are used:
     - ORB keypoint matching (35-40% weight)
     - Shape-based features (30-35% weight)
     - Pixel distribution analysis (25% weight)
     - SSIM comparison (10% weight, if available)
   - Final score is a weighted combination of all methods

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- **Database errors**: Delete `signature_auth.db` to reset the database
- **Signature not matching**: Try adjusting `ORB_SIM_THRESHOLD` in `config.py` (lower for more lenient matching)
- **Port already in use**: Change the port: `uvicorn app:app --port 8001`

## Development

To clear all users from the database:
```bash
python clear_users.py
```

## Security Considerations

- Passwords are hashed using bcrypt with 12 rounds
- Signatures are stored as feature templates, not raw images
- SQL injection protection via SQLAlchemy ORM
- Consider using environment variables for sensitive configuration in production

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

[Your name/contact information]

