"""Acquisition contract tests using a local HTTP server and real ffprobe decoding."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave
from xml.sax.saxutils import escape

from curation import collect


def wav_bytes(seconds=2):
    out = io.BytesIO()
    with wave.open(out, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\0\0' * (seconds * 16000))
    return out.getvalue()


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'archive'
        self.payload = wav_bytes()
        self.requests = []
        self.feeds = {}
        self.interrupted = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append((self.path, dict(self.headers)))
                if self.path in owner.feeds:
                    body = owner.feeds[self.path].encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == '/denied':
                    self.send_error(403)
                    return
                body = b'<html>not audio</html>' if self.path == '/invalid' else owner.payload
                if self.path == '/interrupt' and not owner.interrupted:
                    owner.interrupted = True
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('ETag', '"version1"')
                    self.end_headers()
                    self.wfile.write(body[:1048576])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                header = self.headers.get('Range')
                if header:
                    offset = int(header.removeprefix('bytes=').removesuffix('-'))
                    self.send_response(206)
                    self.send_header('Content-Range', f'bytes {offset}-{len(body)-1}/{len(body)}')
                    body = body[offset:]
                else:
                    self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('ETag', '"version1"')
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.args = SimpleNamespace(root=str(self.root), sources=str(Path(self.tmp.name)/'sources.json'),
                                    max_items=100, max_hours=0, max_source_hours=0,
                                    min_free_gib=0, max_attempts=3, delay=0)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def discovery(self, sources):
        registry = []
        for sid, media_paths in sources.items():
            feed_path = f'/{sid}.rss'
            registry.append({'id':sid,'url':self.base+feed_path,'language':'fil'})
            items = ''.join(f'<item><guid>{sid}-{i}</guid><title>Example {i}</title>'
                            f'<enclosure url="{escape(self.base+path)}" type="audio/wav"/>'
                            '<itunes:duration>9999</itunes:duration></item>' for i,path in enumerate(media_paths))
            self.feeds[feed_path] = '<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>'+items+'</channel></rss>'
        Path(self.args.sources).write_text(json.dumps(registry))
        with contextlib.redirect_stdout(io.StringIO()):
            collect.discover(self.args)

    def run_download(self):
        with patch.object(collect.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            collect.download(self.args)

    def rows(self):
        db = collect.connect(self.root)
        try:
            return [dict(r) for r in db.execute('SELECT * FROM items ORDER BY source_id,id')]
        finally:
            db.close()

    def test_download_measures_audio_and_rediscovery_preserves_completion(self):
        self.discovery({'show':['/audio']})
        self.run_download()
        r = self.rows()[0]
        self.assertEqual(r['status'], 'downloaded')
        self.assertEqual(r['estimated_seconds'], 9999)
        self.assertAlmostEqual(r['duration'], 2)
        self.assertEqual(r['sha256'], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(Path(r['path']).read_bytes(), self.payload)
        self.discovery({'show':['/audio']})
        self.assertEqual(self.rows()[0]['status'], 'downloaded')
        n_requests = len(self.requests)
        self.run_download()
        self.assertEqual(len(self.requests), n_requests)

    def test_interrupted_transfer_resumes_with_validator(self):
        self.payload = wav_bytes(40)
        self.discovery({'show':['/interrupt']})
        self.args.max_items = 1
        self.run_download()
        r = self.rows()[0]
        self.assertEqual(r['status'], 'failed')
        parts = list(self.root.glob('audio/*/*.part'))
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].stat().st_size, 1048576)
        self.run_download()
        r = self.rows()[0]
        self.assertEqual(r['status'], 'downloaded')
        self.assertEqual(Path(r['path']).read_bytes(), self.payload)
        media_requests = [headers for path,headers in self.requests if path == '/interrupt']
        self.assertEqual(media_requests[-1]['Range'], 'bytes=1048576-')
        self.assertEqual(media_requests[-1]['If-Range'], '"version1"')
        self.assertEqual(r['attempts'], 2)

    def test_invalid_media_rejected_and_access_denial_not_retried(self):
        self.discovery({'show':['/invalid','/denied']})
        self.args.max_items = 5
        self.run_download()
        rows = self.rows()
        self.assertEqual({r['status'] for r in rows}, {'failed','unavailable'})
        self.assertFalse(any(r['duration'] for r in rows))
        denied = [r for r in rows if r['url'].endswith('/denied')][0]
        self.assertEqual(denied['attempts'], 1)
        self.assertEqual(len([p for p,_ in self.requests if p == '/denied']), 1)

    def test_fair_order_and_source_caps_use_measured_hours(self):
        self.discovery({'a':['/a1','/a2'], 'b':['/b1','/b2']})
        self.args.max_source_hours = 1/3600
        self.run_download()
        done = [r for r in self.rows() if r['status'] == 'downloaded']
        self.assertEqual(len(done), 2)
        self.assertEqual({r['source_id'] for r in done}, {'a','b'})
        self.assertEqual(len([r for r in self.rows() if r['status'] == 'pending']), 2)
        # Each complete 2-second episode can exceed the 1-second cap once.
        self.assertEqual(sum(r['duration'] for r in done), 4)

    def test_total_cap_stops_after_one_complete_episode(self):
        self.discovery({'a':['/a1','/a2'], 'b':['/b1']})
        self.args.max_hours = 1/3600
        self.run_download()
        self.assertEqual(sum(r['status'] == 'downloaded' for r in self.rows()), 1)

    def test_status_deduplicates_exact_file_bytes(self):
        self.discovery({'a':['/a1'], 'b':['/b1']})
        self.run_download()
        summary = json.loads((self.root/'status.json').read_text())
        unique = summary['exact_file_deduplicated'][0]
        self.assertEqual(unique['unique_files'], 1)
        self.assertAlmostEqual(unique['unique_file_hours'], 2/3600)


if __name__ == '__main__':
    unittest.main()
