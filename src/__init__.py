import sys

# Logs/prints use non-ASCII (→, –, ─); the cp1252 default crashes when output is piped on Windows.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
