from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import APIRouter, Depends, HTTPException
from database.database import db_manager
from database.database_models import users
from features.auth.schemas import UserRegister, UserLogin, Token
from features.auth.passwords import hash_password, verify_password
from features.auth.jwt import create_access_token
from features.auth.dependencies import get_current_user

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/register")
async def register(
    request: UserRegister,
    db: AsyncSession = Depends(db_manager.get_session)
):
    stmt = select(users).where(
        (users.email == request.email) |
        (users.username == request.username)
    )

    result = await db.execute(stmt)
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    new_user = users(
        username=request.username,
        email=request.email,
        hashed_password=hash_password(request.password)
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {
        "message": "User registered successfully"
    }

@router.post("/login", response_model=Token)
async def login(
    request: UserLogin,
    db: AsyncSession = Depends(db_manager.get_session)
):
    stmt = select(users).where(users.email == request.email)

    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        request.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(
        {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@router.get("/me")
async def me(current_user=Depends(get_current_user)):
    return current_user