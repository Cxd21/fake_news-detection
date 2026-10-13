"""
Fake News Detector — Root server entry point.
Forwards to the modular backend in backend/server.py.

Run:  python server.py        then open http://localhost:5000
"""
from backend.server import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
