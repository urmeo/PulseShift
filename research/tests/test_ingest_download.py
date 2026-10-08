from http.client import HTTPResponse
from io import BytesIO
from unittest.mock import Mock

import pytest

from pulseshift.ingest import _download


def _response(body, headers):
    socket = Mock()
    socket.makefile.return_value = BytesIO(
        b"HTTP/1.1 200 OK\r\n" + headers + b"Connection: close\r\n\r\n" + body
    )
    response = HTTPResponse(socket)
    response.begin()
    return response


@pytest.mark.parametrize("headers", [b"Content-Length: 10\r\n", b""])
def test_truncated_download_can_retry_and_cache_complete_response(
    tmp_path, monkeypatch, headers
):
    body = b"0123456789"
    opener = Mock(
        side_effect=[
            _response(b"hi", b"Content-Length: 10\r\n"),
            _response(body, headers),
        ]
    )
    monkeypatch.setattr("pulseshift.ingest.urllib.request.urlopen", opener)
    target = tmp_path / "source.zip"
    url = "https://example.test/source.zip"

    with pytest.raises(RuntimeError, match="received 2 of 10 bytes"):
        _download(url, target)
    assert not target.exists()
    assert not target.with_name("source.zip.part").exists()

    assert _download(url, target) == target
    assert target.read_bytes() == body
    assert not target.with_name("source.zip.part").exists()
    assert _download(url, target) == target
    assert opener.call_count == 2
