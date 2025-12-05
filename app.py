import json
import uuid
from typing import List, Optional

from fastapi import FastAPI, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from auth import hash_password
from config import BASE_DIR, ORB_SIM_THRESHOLD
from database import Base, engine, get_db
from models import User
from signature_utils import (
    compare_signatures,
    preprocess_signature,
    decode_base64_image,
    encode_template,
    verify_signature_against_stored,
)

# Create database tables on startup (simple dev setup)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Signature-Based Auth")

# Static files (CSS / JS) and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory storage for registration sessions
PENDING_REGISTRATIONS = {}  # reg_id -> {"name": ..., "password_hash": ...}


FAKE_PROFILE_PATH = BASE_DIR / "data" / "fake_profiles.json"


def load_fake_profiles() -> List[dict]:
    """Load fake university profile metadata from disk."""
    try:
        with FAKE_PROFILE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data[:10]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    return []


FAKE_PROFILES = load_fake_profiles()


def resolve_fake_profile(db: Session, user_id: int) -> Optional[dict]:
    """Return fake profile data for the user if they are among the first 10 registrations."""
    if not FAKE_PROFILES:
        return None

    rows = (
        db.query(User.id)
        .order_by(User.created_at.asc(), User.id.asc())
        .limit(len(FAKE_PROFILES))
        .all()
    )
    ordered_ids = [row[0] for row in rows]
    try:
        idx = ordered_ids.index(user_id)
    except ValueError:
        return None
    return FAKE_PROFILES[idx]


@app.get("/", response_class=HTMLResponse)
def root():
    """Redirect root to registration page."""
    return RedirectResponse(url="/register")


# ---------------- Registration ----------------

@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    """Initial registration form for username and password."""
    return templates.TemplateResponse(
        "register.html",
        {"request": request},
    )


@app.post("/register", response_class=HTMLResponse)
def register_start(
    request: Request,
    name: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Start registration: validate username, hash password,
    then redirect to signature capture page.
    """
    existing = db.query(User).filter(User.name == name).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Username already exists, please choose another.",
            },
        )

    pwd_hash = hash_password(password)
    reg_id = str(uuid.uuid4())
    PENDING_REGISTRATIONS[reg_id] = {
        "name": name,
        "password_hash": pwd_hash,
    }
    return RedirectResponse(url=f"/register-signature?reg_id={reg_id}", status_code=303)


@app.get("/register-signature", response_class=HTMLResponse)
def register_signature_page(request: Request, reg_id: str):
    """
    Page where the user draws their signature twice for confirmation.
    """
    if reg_id not in PENDING_REGISTRATIONS:
        return RedirectResponse(url="/register")
    return templates.TemplateResponse(
        "register_signature.html",
        {
            "request": request,
            "reg_id": reg_id,
        },
    )


@app.post("/register-signature")
async def register_signature_submit(
    payload: dict,
    db: Session = Depends(get_db),
):
    """
    Receive two signatures from the registration page, compare them,
    and if similarity is above threshold, create a new user.

    Payload:
        {
            "reg_id": "...",
            "sig1": "data:image/png;base64,...",
            "sig2": "data:image/png;base64,..."
        }
    """
    reg_id: Optional[str] = payload.get("reg_id")
    sig1: Optional[str] = payload.get("sig1")
    sig2: Optional[str] = payload.get("sig2")

    if not reg_id or reg_id not in PENDING_REGISTRATIONS:
        return JSONResponse(
            {"ok": False, "error": "Invalid registration session."},
            status_code=400,
        )
    if not sig1 or not sig2:
        return JSONResponse(
            {"ok": False, "error": "Both signatures are required."},
            status_code=400,
        )

    similarity, method = compare_signatures(sig1, sig2)
    threshold = ORB_SIM_THRESHOLD

    if similarity < threshold:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Signatures do not match (similarity={similarity:.3f}, method={method}). Please try again.",
            },
            status_code=400,
        )

    # Use the first signature as the template: compute and store preprocessed image.
    img1 = decode_base64_image(sig1)
    pre1 = preprocess_signature(img1)
    template_json = encode_template(pre1)

    user_info = PENDING_REGISTRATIONS.pop(reg_id)
    user = User(
        name=user_info["name"],
        password_hash=user_info["password_hash"],
        signature_embedding=template_json,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return JSONResponse(
        {
            "ok": True,
            "message": "Registration successful! You can now log in.",
        }
    )


# ---------------- Login ----------------

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    """Login form that asks only for username."""
    return templates.TemplateResponse(
        "login.html",
        {"request": request},
    )


@app.post("/login", response_class=HTMLResponse)
def login_start(
    request: Request,
    name: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Start login: check that the user exists, then redirect to
    the signature-based login page.
    """
    user = db.query(User).filter(User.name == name).first()
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "User not found."},
        )
    return RedirectResponse(
        url=f"/login-signature?username={user.name}", status_code=303
    )


@app.get("/login-signature", response_class=HTMLResponse)
def login_signature_page(request: Request, username: str):
    """
    Page where the user draws their signature for login.
    """
    return templates.TemplateResponse(
        "login_signature.html",
        {"request": request, "username": username},
    )


@app.post("/login-signature")
async def login_signature_submit(
    payload: dict,
    db: Session = Depends(get_db),
):
    """
    Verify a login-time signature against the stored template.

    Payload:
        {
            "username": "...",
            "sig": "data:image/png;base64,..."
        }
    """
    username: Optional[str] = payload.get("username")
    sig: Optional[str] = payload.get("sig")

    if not username or not sig:
        return JSONResponse(
            {"ok": False, "error": "Username and signature are required."},
            status_code=400,
        )

    user = db.query(User).filter(User.name == username).first()
    if not user:
        return JSONResponse({"ok": False, "error": "User not found."}, status_code=404)

    ok, similarity, method = verify_signature_against_stored(
        sig, user.signature_embedding
    )
    threshold = ORB_SIM_THRESHOLD

    if not ok:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Signature verification failed (similarity={similarity:.3f}, method={method}, threshold={threshold}).",
            },
            status_code=401,
        )

    return JSONResponse(
        {
            "ok": True,
            "username": user.name,
            "similarity": similarity,
            "method": method,
        }
    )


@app.get("/user-info", response_class=HTMLResponse)
def user_info_page(request: Request, username: str, db: Session = Depends(get_db)):
    """
    Simple user info page shown after successful login.
    Only displays non-sensitive information.
    """
    user = db.query(User).filter(User.name == username).first()
    if not user:
        return RedirectResponse(url="/login")
    profile_data = resolve_fake_profile(db, user.id)
    return templates.TemplateResponse(
        "user_info.html",
        {
            "request": request,
            "user": user,
            "profile_data": profile_data,
        },
    )


