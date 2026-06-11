#!/usr/bin/env bash

# Hamlin Software | AI Model Tuner & Evaluator Shutdown Script
# Automatically stops the background application.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}================================================================${NC}"
echo -e "${BOLD}${CYAN}       Hamlin Software | AI Model Tuner & Evaluator Shutdown    ${NC}"
echo -e "${BOLD}${CYAN}================================================================${NC}"
echo ""

PID_FILE=".app.pid"
STOPPED=false

# Method 1: Check PID file
if [ -f "$PID_FILE" ]; then
    APP_PID=$(cat "$PID_FILE")
    echo -e "${BLUE}Found running application process ID (PID: $APP_PID) in $PID_FILE...${NC}"
    if kill -0 "$APP_PID" 2>/dev/null; then
        echo -e "${YELLOW}Stopping process $APP_PID...${NC}"
        kill "$APP_PID" 2>/dev/null
        
        # Wait up to 5 seconds for clean exit
        for i in {1..5}; do
            if ! kill -0 "$APP_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$APP_PID" 2>/dev/null; then
            echo -e "${YELLOW}Process did not exit, forcing shutdown...${NC}"
            kill -9 "$APP_PID" 2>/dev/null
        fi
        
        echo -e "${GREEN}${BOLD}✓ Application stopped successfully.${NC}"
        STOPPED=true
    else
        echo -e "${YELLOW}Process ID $APP_PID in $PID_FILE was not running.${NC}"
    fi
    rm -f "$PID_FILE"
fi

# Method 2: Check port 5000 as a backup fallback
if command -v lsof &> /dev/null; then
    PORT_PIDS=$(lsof -t -i:5000 2>/dev/null)
    if [ -n "$PORT_PIDS" ]; then
        echo -e "${BLUE}Found processes listening on port 5000...${NC}"
        for PORT_PID in $PORT_PIDS; do
            echo -e "${YELLOW}Stopping process $PORT_PID...${NC}"
            kill "$PORT_PID" 2>/dev/null
            sleep 0.5
            if kill -0 "$PORT_PID" 2>/dev/null; then
                kill -9 "$PORT_PID" 2>/dev/null
            fi
        done
        echo -e "${GREEN}${BOLD}✓ Port 5000 cleared.${NC}"
        STOPPED=true
    fi
fi

if [ "$STOPPED" = true ]; then
    echo -e "${GREEN}${BOLD}✓ Service shutdown completed.${NC}"
else
    echo -e "${YELLOW}No running application or service on port 5000 could be found.${NC}"
fi
