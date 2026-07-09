#!/usr/bin/env bash

# Hamlin Software | AI Model Tuner & Evaluator Setup Script
# Automatically detects environment, configures dependencies, and starts the application in the background.

# Set color codes for beautiful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}================================================================${NC}"
echo -e "${BOLD}${CYAN}       Hamlin Software | AI Model Tuner & Evaluator Setup       ${NC}"
echo -e "${BOLD}${CYAN}================================================================${NC}"
echo ""

# Helper function to launch the app in the background and verify startup
launch_app_background() {
    local interpreter="$1"
    
    echo -e "${BLUE}Launching application in the background...${NC}"
    nohup $interpreter app.py > app.log 2>&1 &
    local pid=$!
    echo $pid > .app.pid
    
    # Wait to ensure it didn't crash immediately (e.g., port already bound or import errors)
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        echo -e "${GREEN}${BOLD}✓ Application successfully started in the background (PID: $pid)!${NC}"
        echo -e "${GREEN}Server logs are redirected to: ${BOLD}app.log${NC}"
        echo ""
        echo -e "${BOLD}${CYAN}Application link: http://127.0.0.1:${SERVICE_PORT:-5000}${NC}"
        echo -e "${YELLOW}To stop the application, run: ./shutdown.sh${NC}"
        echo ""
        exit 0
    else
        echo -e "${RED}${BOLD}Error: Application failed to start.${NC}"
        echo -e "${RED}Last few lines of app.log:${NC}"
        tail -n 15 app.log
        rm -f .app.pid
        exit 1
    fi
}

# Step 1: Find and verify base Python 3 executable (bypassing any active virtual environments that might be deleted/rebuilt)
PYTHON_CMD="python3"
if [ -n "$VIRTUAL_ENV" ]; then
    # Strip the virtual environment's bin directory from PATH to locate the system python3
    CLEAN_PATH=$(echo "$PATH" | sed -e "s|${VIRTUAL_ENV}/bin:||g" -e "s|:${VIRTUAL_ENV}/bin||g")
    SYSTEM_PYTHON=$(PATH="$CLEAN_PATH" which python3 2>/dev/null)
    if [ -n "$SYSTEM_PYTHON" ] && [ -x "$SYSTEM_PYTHON" ]; then
        PYTHON_CMD="$SYSTEM_PYTHON"
    else
        # Fallback to standard absolute system location
        if [ -x "/usr/bin/python3" ]; then
            PYTHON_CMD="/usr/bin/python3"
        fi
    fi
else
    # Verify the current python3 in PATH is executable
    CURRENT_PYTHON=$(which python3 2>/dev/null)
    if [ -n "$CURRENT_PYTHON" ] && [ -x "$CURRENT_PYTHON" ]; then
        PYTHON_CMD="$CURRENT_PYTHON"
    else
        # Fallback to standard absolute system location
        if [ -x "/usr/bin/python3" ]; then
            PYTHON_CMD="/usr/bin/python3"
        fi
    fi
fi

# Ensure the resolved python3 executable is valid and works
if ! [ -x "$PYTHON_CMD" ]; then
    echo -e "${RED}${BOLD}Error: No valid python3 executable could be found.${NC}"
    echo -e "Please ensure Python 3.12+ is installed and accessible."
    exit 1
fi

# Step 2: Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

echo -e "${BLUE}Detected Python version: ${BOLD}$PYTHON_VERSION${NC} (${PYTHON_CMD})"

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
    echo -e "${YELLOW}${BOLD}Warning: Application is designed for Python 3.12+. Found Python $PYTHON_VERSION.${NC}"
    echo -e "We will attempt to proceed, but some libraries or features may fail."
fi

# Step 3: Check if dependencies are already satisfied in the resolved Python environment
echo -e "${BLUE}Checking dependencies (Flask, scikit-learn, pandas, numpy)...${NC}"
if $PYTHON_CMD -c "import flask, sklearn, pandas, numpy" &> /dev/null; then
    echo -e "${GREEN}${BOLD}✓ All required dependencies are already installed!${NC}"
    launch_app_background "$PYTHON_CMD"
fi

# Step 4: Try creating a virtual environment if dependencies are not satisfied
echo -e "${YELLOW}Dependencies not fully satisfied. Setting up environment...${NC}"

# If .venv exists but is incomplete, remove it to start clean
if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
    echo -e "${YELLOW}Found incomplete .venv directory, cleaning up...${NC}"
    rm -rf .venv
fi

USE_VENV=false
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}Attempting to create a virtual environment in .venv...${NC}"
    if $PYTHON_CMD -m venv .venv 2>&1; then
        echo -e "${GREEN}✓ Virtual environment created successfully.${NC}"
        USE_VENV=true
    else
        echo -e "${YELLOW}Warning: Failed to create virtual environment via python3 -m venv.${NC}"
        echo -e "This is often because python3-venv / ensurepip is not installed."
        echo -e "Attempting fallback to user-space package installation...${NC}"
    fi
else
    echo -e "${GREEN}Found existing .venv directory.${NC}"
    USE_VENV=true
fi

# Step 5: Install dependencies in venv
if [ "$USE_VENV" = true ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source .venv/bin/activate
    
    echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
    if pip install -r requirements.txt; then
        echo -e "${GREEN}${BOLD}✓ Dependencies successfully installed in virtual environment!${NC}"
        launch_app_background "python3"
    else
        echo -e "${RED}${BOLD}Error: Failed to install dependencies in virtual environment.${NC}"
        deactivate
    fi
fi

# Step 6: Fallback to user-space installation if virtual environment failed or pip install failed
echo -e "${BLUE}Attempting to install requirements globally/user-space using --break-system-packages...${NC}"
if $PYTHON_CMD -m pip install --user --break-system-packages -r requirements.txt; then
    echo -e "${GREEN}${BOLD}✓ Dependencies successfully installed in user-space!${NC}"
    launch_app_background "$PYTHON_CMD"
else
    echo -e "${RED}${BOLD}Critical Error: Failed to install required dependencies.${NC}"
    echo -e "Please install the following packages manually:"
    echo -e "  - flask"
    echo -e "  - scikit-learn"
    echo -e "  - pandas"
    echo -e "  - numpy"
    exit 1
fi
