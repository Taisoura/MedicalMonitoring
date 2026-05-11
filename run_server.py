"""
Launch the Medical Monitoring API server.

Usage:
    python run_server.py [--host HOST] [--port PORT] [--reload]
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Medical Monitoring API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"Starting Medical Monitoring API on http://{args.host}:{args.port}")
    print(f"  Docs: http://localhost:{args.port}/docs")
    print(f"  Health: http://localhost:{args.port}/health")

    uvicorn.run(
        "medical_monitoring.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
