import http.server
import socketserver
import mimetypes


# Custom handler to add proper headers for WebAssembly and gzip
class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        if self.path.endswith(".wasm"):
            # Add correct MIME type for WebAssembly files
            self.send_header("Content-Type", "application/wasm")
        if self.path.endswith(".gz"):
            # Add Content-Encoding for gzipped files
            self.send_header("Content-Encoding", "gzip")
        super().end_headers()


# Start the server on port 8000
PORT = 8000
Handler = CustomHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), Handler)

print(f"Serving at http://localhost:{PORT}")
httpd.serve_forever()
