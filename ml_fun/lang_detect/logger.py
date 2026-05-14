import logging
import logging.handlers
import os
import pickle
import socketserver
import struct
import threading

from ml_fun.lang_detect.data import DATA_DIR

DEFAULT_LOG_PORT = 9020


class _RankFilter(logging.Filter):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "rank"):
            record.rank = self.rank
        return True


class _LogRecordStreamHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            record = logging.makeLogRecord(pickle.loads(chunk))
            logging.getLogger(record.name).handle(record)


class _LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _start_log_server(port: int) -> None:
    server = _LogRecordSocketReceiver(("0.0.0.0", port), _LogRecordStreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def setup_logging(timestamp: str, rank: int) -> None:
    logs_dir = DATA_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"logs_{timestamp}_rank{rank}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] rank=%(rank)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    rank_filter = _RankFilter(rank)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(rank_filter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(rank_filter)
    root.addHandler(file_handler)

    log_port = int(os.environ.get("LOG_PORT", DEFAULT_LOG_PORT))
    if rank == 0:
        _start_log_server(log_port)
    else:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        socket_handler = logging.handlers.SocketHandler(master_addr, log_port)
        socket_handler.addFilter(rank_filter)
        root.addHandler(socket_handler)
