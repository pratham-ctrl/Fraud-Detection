import streamlit as st
import hashlib
from sqlalchemy.orm import Session
from models import User, get_db
from sqlalchemy.exc import IntegrityError, OperationalError
from contextlib import contextmanager
import re

@contextmanager
def db_session():
    """Context manager for database sessions with error handling"""
    db = next(get_db())
    try:
        yield db
    except OperationalError as e:
        db.rollback()
        st.error("Database connection error. Please try again.")
        raise
    finally:
        db.close()

def get_current_db():
    """Get database session with error handling"""
    try:
        return next(get_db())
    except Exception as e:
        st.error("Could not connect to database. Please try again.")
        raise

def initialize_auth():
    """Initialize authentication - no longer needed with database"""
    pass

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email):
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def is_valid_password(password):
    """Validate password requirements"""
    return len(password) >= 8

def create_user(email, password):
    """Create a new user in the database"""
    try:
        # Validate email format
        if not is_valid_email(email):
            st.error("Please enter a valid email address.")
            return False

        # Validate password length
        if not is_valid_password(password):
            st.error("Password must be at least 8 characters long.")
            return False

        with db_session() as db:
            hashed_password = hash_password(password)
            user = User(
                email=email,  # Changed from username to email
                password_hash=hashed_password
            )
            db.add(user)
            db.commit()
            return True
    except IntegrityError:
        st.error("Email already registered. Please use a different email address.")
        return False
    except Exception as e:
        st.error(f"An error occurred during registration. Please try again.")
        return False

def check_password(email, password):
    """Verify user credentials against database"""
    try:
        with db_session() as db:
            user = db.query(User).filter(User.email == email).first()  # Changed from username to email
            if not user:
                return False
            return user.password_hash == hash_password(password)
    except Exception as e:
        st.error("Could not verify credentials. Please try again.")
        return False

def get_user_id(email):  # Changed from username to email
    """Get user ID from email"""
    try:
        with db_session() as db:
            user = db.query(User).filter(User.email == email).first()
            return user.id if user else None
    except Exception as e:
        st.error("Could not retrieve user information. Please try again.")
        return None