"""AIM Realtime Voice — FastAPI entry point.

Exposes the OpenAI Realtime WebSocket endpoint Open WebUI 0.12's voice bridge
connects to (`/v1/realtime?model=...`), plus /healthz. The protocol logic and
upstream orchestration live in session.py.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from config import APP_VERSION, Config
from session import Call
from upstream import Upstreams

CFG = Config()
logging.basicConfig(level=getattr(logging, CFG.log_level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aimrealtime")

UPSTREAMS = Upstreams(CFG)
ACTIVE_CALLS = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    await UPSTREAMS.start()
    log.info("aimrealtime %s ready %s", APP_VERSION, CFG.masked())
    yield
    await UPSTREAMS.close()


app = FastAPI(title="AIM Realtime Voice", version=APP_VERSION, lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    resolved = UPSTREAMS.router.endpoint
    base, headers = resolved if resolved else ("(unresolved)", {})
    return JSONResponse(
        {
            "status": "ok",
            "version": APP_VERSION,
            "active_calls": ACTIVE_CALLS,
            "router": base,
            "router_host": headers.get("Host", ""),
            "config": CFG.masked(),
        }
    )


@app.get("/v1/models")
async def models() -> JSONResponse:
    # llm-init (Model Console) waits for exactly this id on the engine's
    # /v1/models before it declares the model ready to the Olares Router.
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": CFG.advertised_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "aimighty",
                }
            ],
        }
    )


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket) -> None:
    global ACTIVE_CALLS
    # The entrance Host tells us this machine's base domain, which is how the
    # Router is found when the platform hosts file is not mounted.
    UPSTREAMS.request_host = ws.headers.get("host", "")
    if CFG.api_key:
        if ws.headers.get("authorization", "") != f"Bearer {CFG.api_key}":
            await ws.close(code=4401)
            return
    if ACTIVE_CALLS >= CFG.max_calls:
        await ws.close(code=4429)
        return
    call = Call(ws, ws.query_params.get("model", ""), CFG, UPSTREAMS)
    ACTIVE_CALLS += 1
    try:
        await call.run()
    except Exception:
        log.debug("call ended with exception", exc_info=True)
    finally:
        await call.close()
        ACTIVE_CALLS -= 1
        log.info("call closed session=%s active=%d", call.session_id, ACTIVE_CALLS)
