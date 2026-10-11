"""Dev only: print an access JWT that this API accepts.

This service has no login endpoint (security.py only verifies tokens),
so for local testing mint one with the same secret from .env.

Run from the rag-service/ folder:
    python dev_token.py            # user "dev-user", valid 24h
    python dev_token.py alice      # user "alice"
"""

import sys
import time

from dotenv import load_dotenv
from jose import jwt

# Same lookup as main.py: finds .env here or in a parent folder (e.g. the project root).
load_dotenv()

from app.core.config import settings  # noqa: E402


user_id = sys.argv[1] if len(sys.argv) > 1 else "dev-user"

claims = {
    "sub": user_id,
    "type": "access",
    "exp": int(time.time()) + 24 * 60 * 60,
}

print(jwt.encode(claims, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM))
