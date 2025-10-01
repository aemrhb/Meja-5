#!/bin/bash

# Cache cleaning script for MEJA project
# Run with: bash clean_cache.sh [option]
# Options: logs, checkpoints, python, all, safe

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration paths
LOG_DIR="/bigwork/nhgnheid/log_nomeformer/log1-7"
CHECKPOINT_DIR="/bigwork/nhgnheid/Ex_meja_downsteram/Ex_delf/Ex_3"
BASE_DIR="/bigwork/nhgnheid"

echo -e "${GREEN}MEJA Cache Cleaning Script${NC}"
echo "=================================="

clean_logs() {
    echo -e "${YELLOW}Cleaning log directory...${NC}"
    if [ -d "$LOG_DIR" ]; then
        # Remove old log files (older than 7 days)
        find "$LOG_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null || true
        # Remove TensorBoard event files
        find "$LOG_DIR" -name "events.out.tfevents.*" -delete 2>/dev/null || true
        # Remove temporary files
        find "$LOG_DIR" -name "*.tmp" -delete 2>/dev/null || true
        echo -e "${GREEN}✓ Log directory cleaned${NC}"
    else
        echo -e "${RED}✗ Log directory not found: $LOG_DIR${NC}"
    fi
}

clean_checkpoints() {
    echo -e "${YELLOW}Cleaning checkpoint directory...${NC}"
    if [ -d "$CHECKPOINT_DIR" ]; then
        cd "$CHECKPOINT_DIR"
        # Count current .pth files
        pth_count=$(ls -1 *.pth 2>/dev/null | wc -l)
        if [ $pth_count -gt 5 ]; then
            echo "Found $pth_count checkpoint files. Keeping only the 5 most recent..."
            # Keep only the 5 most recent .pth files
            ls -t *.pth | tail -n +6 | xargs rm -f
            echo -e "${GREEN}✓ Old checkpoints removed${NC}"
        else
            echo "Only $pth_count checkpoint files found. No cleanup needed."
        fi
    else
        echo -e "${RED}✗ Checkpoint directory not found: $CHECKPOINT_DIR${NC}"
    fi
}

clean_python_cache() {
    echo -e "${YELLOW}Cleaning Python cache files...${NC}"
    # Remove __pycache__ directories
    find "$BASE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    # Remove .pyc and .pyo files
    find "$BASE_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$BASE_DIR" -name "*.pyo" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Python cache cleaned${NC}"
}

clean_system_cache() {
    echo -e "${YELLOW}Cleaning system cache files...${NC}"
    # Remove common system cache files
    find "$BASE_DIR" -name ".DS_Store" -delete 2>/dev/null || true
    find "$BASE_DIR" -name "Thumbs.db" -delete 2>/dev/null || true
    find "$BASE_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$BASE_DIR" -name "*.temp" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ System cache cleaned${NC}"
}

clear_gpu_cache() {
    echo -e "${YELLOW}Clearing GPU cache...${NC}"
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('GPU cache cleared')
    else:
        print('CUDA not available')
except ImportError:
    print('PyTorch not available')
" 2>/dev/null || echo "Could not clear GPU cache"
    echo -e "${GREEN}✓ GPU cache clearing attempted${NC}"
}

show_disk_usage() {
    echo -e "${YELLOW}Disk usage for main directories:${NC}"
    for dir in "$LOG_DIR" "$CHECKPOINT_DIR"; do
        if [ -d "$dir" ]; then
            echo "$(du -sh "$dir" 2>/dev/null || echo "0B $dir")"
        fi
    done
}

# Main execution
case "${1:-safe}" in
    "logs")
        clean_logs
        ;;
    "checkpoints")
        clean_checkpoints
        ;;
    "python")
        clean_python_cache
        clear_gpu_cache
        ;;
    "system")
        clean_system_cache
        ;;
    "all")
        echo -e "${RED}WARNING: This will clean ALL cache types!${NC}"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            clean_logs
            clean_checkpoints
            clean_python_cache
            clean_system_cache
            clear_gpu_cache
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    "safe")
        echo -e "${GREEN}Running safe cleanup (Python cache + system cache only)${NC}"
        clean_python_cache
        clean_system_cache
        clear_gpu_cache
        ;;
    "usage")
        show_disk_usage
        ;;
    *)
        echo "Usage: $0 [logs|checkpoints|python|system|all|safe|usage]"
        echo ""
        echo "Options:"
        echo "  logs        - Clean log files older than 7 days"
        echo "  checkpoints - Keep only 5 most recent checkpoint files"
        echo "  python      - Clean Python cache and GPU memory"
        echo "  system      - Clean system cache files (.DS_Store, etc.)"
        echo "  all         - Clean everything (requires confirmation)"
        echo "  safe        - Clean Python and system cache (default)"
        echo "  usage       - Show disk usage of main directories"
        exit 1
        ;;
esac

echo -e "${GREEN}Cache cleaning completed!${NC}"
show_disk_usage





