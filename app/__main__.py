import os

import uvicorn


def main():
    host = os.getenv("APP_HOST") or os.getenv("CLASSROOM_UI_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT") or os.getenv("CLASSROOM_UI_PORT", "8000"))
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
