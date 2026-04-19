#!/bin/bash
#
# Model Export Tool for Offline Deployment
# Interactive version - KISS principle
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

OUTPUT_DIR="./models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print header
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}FunASR-API Model Export Tool${NC}                        ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     Export models for offline deployment                 ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
echo -e "${BLUE}Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ python3 not found${NC}"
    exit 1
fi

# Check if running from project root or scripts directory
if [ -f "$PROJECT_ROOT/app/utils/download_models.py" ]; then
    cd "$PROJECT_ROOT"
elif [ -f "$SCRIPT_DIR/../app/utils/download_models.py" ]; then
    cd "$SCRIPT_DIR/.."
else
    echo -e "${RED}✗ Cannot find app/utils/download_models.py${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

echo -e "${GREEN}✓ Python OK${NC}"
echo ""

# Confirm
echo -e "${YELLOW}Export settings:${NC}"
echo -e "  Models: ${CYAN}Current runtime plan (auto-selected Qwen + realtime stack)${NC}"
echo -e "  Output: ${CYAN}${OUTPUT_DIR}/${NC}"
echo ""
read -p "Start export? [Y/n]: " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Export
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Exporting models...${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Remove existing models dir to ensure clean state
rm -rf "${OUTPUT_DIR}"

# Run Python export
python3 -m app.utils.download_models --export-dir "${OUTPUT_DIR}"

echo ""

# Package
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Packaging...${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

PACKAGE="funasr-models-$(date +%Y%m%d-%H%M).tar.gz"

# Use pigz for multi-threaded compression if available
if command -v pigz &> /dev/null; then
    echo "Using pigz for multi-threaded compression..."
    # Get CPU cores (cross-platform: Linux and macOS)
    if command -v nproc &> /dev/null; then
        CPU_CORES=$(nproc)
    elif command -v sysctl &> /dev/null; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES=4
    fi
    tar -cf - "${OUTPUT_DIR}" | pigz -p "${CPU_CORES}" > "${PACKAGE}"
else
    echo "pigz not found, using standard gzip..."
    tar -czf "${PACKAGE}" "${OUTPUT_DIR}"
fi

SIZE=$(du -sh "${PACKAGE}" | cut -f1)

# Done
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Export Complete!                                          ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Package: ${CYAN}${PACKAGE}${NC}"
echo -e "Size:    ${CYAN}${SIZE}${NC}"
echo ""
echo -e "${YELLOW}To deploy on an offline server:${NC}"
echo ""
echo -e "1. Copy package:  ${CYAN}scp ${PACKAGE} user@server:/opt/funasr-api/${NC}"
echo -e "2. Extract:       ${CYAN}tar -xzvf ${PACKAGE}${NC}"
echo -e "3. Start service: ${CYAN}docker-compose up -d${NC}"
echo ""
