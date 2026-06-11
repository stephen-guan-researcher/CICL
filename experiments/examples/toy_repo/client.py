class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path


def retryable_status(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}

