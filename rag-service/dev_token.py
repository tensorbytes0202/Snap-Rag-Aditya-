"""Dev only: print an access JWT that this API accepts.

This service has no login endpoint (security.py only verifies tokens),
so for local testing mint one with the same secret from .env.

Run from the rag-service/ folder:
    python dev_token.py            # user "dev-user", valid 24h
    python dev_token.py alice      # user "alice"
"""

import sys

from dotenv import load_dotenv

# Same lookup as main.py: finds .env here or in a parent folder (e.g. the project root).
load_dotenv()

from app.core.security import create_access_token  # noqa: E402


user_id = sys.argv[1] if len(sys.argv) > 1 else "dev-user"

print(create_access_token(user_id, 24))
