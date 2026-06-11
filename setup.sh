#!/usr/bin/env bash

# Hamlin Software | AI Model Tuner & Evaluator Setup Script
# Automatically detects environment, configures dependencies, and starts the application.

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

# Step 1: Verify Python 3 installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}${BOLD}Error: python3 is not installed on this system.${NC}"
    echo -e "Please install Python 3.12+ and try again."
    exit 1
fi

# Step 2: Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

echo -e "${BLUE}Detected Python version: ${BOLD}$PYTHON_VERSION${NC}"

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
    echo -e "${YELLOW}${BOLD}Warning: Application is designed for Python 3.12+. Found Python $PYTHON_VERSION.${NC}"
    echo -e "We will attempt to proceed, but some libraries or features may fail."
fi

# Step 3: Check if dependencies are already satisfied in the current Python environment
echo -e "${BLUE}Checking dependencies (Flask, scikit-learn, pandas, numpy)...${NC}"
if python3 -c "import flask, sklearn, pandas, numpy" &> /dev/null; then
    echo -e "${GREEN}${BOLD}✓ All required dependencies are already installed!${NC}"
    echo -e "${GREEN}Starting application using system/user Python interpreter...${NC}"
    echo ""
    echo -e "${BOLD}${CYAN}Application link: http://127.0.0.1:5000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the application.${NC}"
    echo ""
    exec python3 app.py
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
    if python3 -m venv .venv 2>&1; then
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

# Step 5: Install dependencies
if [ "$USE_VENV" = true ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source .venv/bin/activate
    
    echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
    if pip install -r requirements.txt; then
        echo -e "${GREEN}${BOLD}✓ Dependencies successfully installed in virtual environment!${NC}"
        echo -e "${GREEN}Starting application in virtual environment...${NC}"
        echo ""
        echo -e "${BOLD}${CYAN}Application link: http://127.0.0.1:5000${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the application.${NC}"
        echo ""
        exec python3 app.py
    else
        echo -e "${RED}${BOLD}Error: Failed to install dependencies in virtual environment.${NC}"
        deactivate
    fi
fi

# Step 6: Fallback to user-space installation if virtual environment failed or pip install failed
echo -e "${BLUE}Attempting to install requirements globally/user-space using --break-system-packages...${NC}"
if python3 -m pip install --user --break-system-packages -r requirements.txt; then
    echo -e "${GREEN}${BOLD}✓ Dependencies successfully installed in user-space!${NC}"
    echo -e "${GREEN}Starting application...${NC}"
    echo ""
    echo -e "${BOLD}${CYAN}Application link: http://127.0.0.1:5000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the application.${NC}"
    echo ""
    exec python3 app.py
else
    echo -e "${RED}${BOLD}Critical Error: Failed to install required dependencies.${NC}"
    echo -e "Please install the following packages manually:"
    echo -e "  - flask"
    echo -e "  - scikit-learn"
    echo -e "  - pandas"
    echo -e "  - numpy"
    exit 1
fi
