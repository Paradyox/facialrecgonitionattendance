"""
Simple HTTP server to handle points updates from dashboard
Receives POST requests from classroom_dashboard.html and saves points to JSON
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointsHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST request to save points"""
        if self.path == '/save_points':
            try:
                # Read the request body
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode('utf-8'))
                
                # Save to student_points.json
                points_file = Path("student_points.json")
                with open(points_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"✓ Points saved: {len(data.get('points', {}))} students")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                
            except Exception as e:
                logger.error(f"Error saving points: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

if __name__ == '__main__':
    PORT = 8001
    server = HTTPServer(('localhost', PORT), PointsHandler)
    logger.info(f"Points server running on http://localhost:{PORT}")
    logger.info("Listening for point updates from dashboard...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nServer stopped")
