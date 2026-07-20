
import http.server, os
os.chdir(r"C:\Users\Administrator\Documents\ai agent\templates")
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()
http.server.HTTPServer(("0.0.0.0", 5011), H).serve_forever()
