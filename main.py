"""
main.py - Entry point để chạy RAG Chatbot API server
"""

import uvicorn

if __name__ == "__main__":
    # Import app từ api/app.py
    from api.app import app
    
    print("=" * 60)
    print("  RAG Chatbot API Server")
    print("=" * 60)
    print("\n  🚀 Starting server at http://localhost:8000")
    print("  📍 API docs: http://localhost:8000/docs")
    print("  💬 Chat interface: http://localhost:8000/\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
