from dotenv import load_dotenv

# LangSmith reads LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT
# straight from os.environ, but pydantic-settings only parses .env into
# `settings` without exporting it. Load .env into the environment first,
# before any module that is traced gets imported.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configures logging (level from LOG_LEVEL, default INFO) as a side
# effect of import. Imported first and explicitly -- rather than left
# to whichever service module happens to import it transitively -- so
# it is deterministically configured once at process startup. Uses
# logging.basicConfig, which only ever touches the root logger's own
# handlers/format; it never reconfigures or disables uvicorn's
# "uvicorn"/"uvicorn.error"/"uvicorn.access" loggers, which manage
# their own handlers independently.
import app.core.logger  # noqa: F401

from app.core.lifespan import lifespan
from app.api.routes.query import router as query_router
from app.api.routes.upload import router as upload_router
from app.api.routes.chats import router as chats_router
from app.api.routes.auth import router as auth_router


app = FastAPI(
    title="RAG Service",
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ==========================================================
# ROUTES
# ==========================================================

app.include_router(
    query_router
)

app.include_router(
    upload_router
)

app.include_router(
    chats_router
)

app.include_router(
    auth_router
)


# ==========================================================
# Health Check
# ==========================================================

@app.get("/")
def home():

    return {
        "service": "rag",
        "status": "running",
    }
