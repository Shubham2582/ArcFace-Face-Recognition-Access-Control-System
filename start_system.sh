#!/bin/bash
# Startup script to run both API and WhatsApp bot

echo "🚀 Starting Face Recognition System..."
echo ""

# Check if Python virtual environment exists
# if [ ! -d ".venv" ]; then
#     echo "❌ Virtual environment not found. Please create one first."
#     echo "Run: python -m venv .venv"
#     exit 1
# fi

# Check if Node modules are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
    echo ""
fi

# Start the Flask API in background
echo "🐍 Starting Flask API server..."
source .venv/bin/activate
python api_service.py &
API_PID=$!
echo "✅ API started (PID: $API_PID)"
echo ""

# Wait a bit for API to start
echo "⏳ Waiting for API to initialize..."
sleep 5
echo ""

# Start the WhatsApp bot
echo "📱 Starting WhatsApp Bot..."
node whatsapp_bot.js &
BOT_PID=$!
echo "✅ WhatsApp Bot started (PID: $BOT_PID)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ System is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 API:        http://localhost:5000"
echo "📱 WhatsApp:   Scan QR code above"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services..."
echo ""

# Function to handle Ctrl+C
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID 2>/dev/null
    kill $BOT_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
