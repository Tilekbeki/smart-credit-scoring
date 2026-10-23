"""HTTP adapter for the versioned, stateless scoring contract."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

from scoring.models import get_model
from scoring.service import ValidationError, score_application


MAX_BODY_BYTES = 16_384


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "service": "scoring"})
        else:
            self._json(404, {"error": "not_found", "message": "Маршрут не найден."})

    def do_POST(self):
        if self.path != "/v1/score":
            self._json(404, {"error": "not_found", "message": "Маршрут не найден."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "unsupported_media_type", "message": "Требуется application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(400, {"error": "invalid_content_length", "message": "Укажите длину JSON-тела."})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            status = 413 if length > MAX_BODY_BYTES else 400
            self._json(status, {"error": "invalid_body_size", "message": "Недопустимый размер JSON-тела."})
            return
        try:
            data = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._json(400, {"error": "invalid_json", "message": "Некорректный JSON."})
            return
        try:
            result = score_application(data)
        except ValidationError as error:
            self._json(422, {"error": "validation_error", "message": str(error)})
            return
        except Exception:
            self.log_error("scoring model failed")
            self._json(500, {"error": "scoring_error", "message": "Не удалось рассчитать скоринговый балл."})
            return
        self._json(200, result)

    def _json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    get_model()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
