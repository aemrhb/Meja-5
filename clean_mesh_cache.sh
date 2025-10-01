#!/bin/bash

# Mesh Cache Cleaning Script for MEJA Dataset
# This script cleans the mesh preprocessing cache created by MeshDataset and MeshTextureDataset
# Usage: bash clean_mesh_cache.sh [option]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration paths from your config
TRAIN_MESH_DIR="/bigwork/nhgnheid/data_Dlft/trian_mesh"
VAL_MESH_DIR="/bigwork/nhgnheid/data_Dlft/test_mesh"
TEXTURE_TRAIN_DIR="/bigwork/nhgnheid/data_Dlft//train_texture"
TEXTURE_VAL_DIR="//bigwork/nhgnheid/data_Dlft//test_texture"

# Cache directory names (as defined in your dataset classes)
CLUSTER_CACHE_DIR=".cluster_cache"
CLUSTER_TEXTURE_CACHE_DIR=".cluster_texture_cache"

echo -e "${GREEN}MEJA Mesh Cache Cleaning Script${NC}"
echo "========================================="

show_cache_info() {
    echo -e "${BLUE}Cache Information:${NC}"
    echo "==================="
    
    for mesh_dir in "$TRAIN_MESH_DIR" "$VAL_MESH_DIR"; do
        if [ -d "$mesh_dir" ]; then
            echo -e "${YELLOW}Directory: $mesh_dir${NC}"
            
            # Check cluster cache
            cluster_cache="$mesh_dir/$CLUSTER_CACHE_DIR"
            if [ -d "$cluster_cache" ]; then
                cache_size=$(du -sh "$cluster_cache" 2>/dev/null | cut -f1)
                file_count=$(find "$cluster_cache" -name "*.pkl" 2>/dev/null | wc -l)
                echo "  └── Cluster cache: $cache_size ($file_count .pkl files)"
            else
                echo "  └── Cluster cache: Not found"
            fi
            
            # Check texture cache
            texture_cache="$mesh_dir/$CLUSTER_TEXTURE_CACHE_DIR"
            if [ -d "$texture_cache" ]; then
                cache_size=$(du -sh "$texture_cache" 2>/dev/null | cut -f1)
                file_count=$(find "$texture_cache" -name "*.pkl" 2>/dev/null | wc -l)
                echo "  └── Texture cache: $cache_size ($file_count .pkl files)"
            else
                echo "  └── Texture cache: Not found"
            fi
            echo
        else
            echo -e "${RED}Directory not found: $mesh_dir${NC}"
        fi
    done
}

clean_cluster_cache() {
    echo -e "${YELLOW}Cleaning cluster cache files...${NC}"
    
    for mesh_dir in "$TRAIN_MESH_DIR" "$VAL_MESH_DIR"; do
        cluster_cache="$mesh_dir/$CLUSTER_CACHE_DIR"
        if [ -d "$cluster_cache" ]; then
            echo "Cleaning: $cluster_cache"
            rm -rf "$cluster_cache"/*
            echo -e "${GREEN}✓ Cluster cache cleaned for $(basename "$mesh_dir")${NC}"
        else
            echo -e "${YELLOW}No cluster cache found in $(basename "$mesh_dir")${NC}"
        fi
    done
}

clean_texture_cache() {
    echo -e "${YELLOW}Cleaning texture cache files...${NC}"
    
    for mesh_dir in "$TRAIN_MESH_DIR" "$VAL_MESH_DIR"; do
        texture_cache="$mesh_dir/$CLUSTER_TEXTURE_CACHE_DIR"
        if [ -d "$texture_cache" ]; then
            echo "Cleaning: $texture_cache"
            rm -rf "$texture_cache"/*
            echo -e "${GREEN}✓ Texture cache cleaned for $(basename "$mesh_dir")${NC}"
        else
            echo -e "${YELLOW}No texture cache found in $(basename "$mesh_dir")${NC}"
        fi
    done
}

clean_old_cache() {
    echo -e "${YELLOW}Cleaning old cache files (older than 7 days)...${NC}"
    
    for mesh_dir in "$TRAIN_MESH_DIR" "$VAL_MESH_DIR"; do
        for cache_type in "$CLUSTER_CACHE_DIR" "$CLUSTER_TEXTURE_CACHE_DIR"; do
            cache_path="$mesh_dir/$cache_type"
            if [ -d "$cache_path" ]; then
                old_files=$(find "$cache_path" -name "*.pkl" -mtime +7 2>/dev/null)
                if [ -n "$old_files" ]; then
                    echo "$old_files" | xargs rm -f
                    echo -e "${GREEN}✓ Removed old files from $cache_type in $(basename "$mesh_dir")${NC}"
                else
                    echo "No old files found in $cache_type in $(basename "$mesh_dir")"
                fi
            fi
        done
    done
}

clean_corrupted_cache() {
    echo -e "${YELLOW}Cleaning potentially corrupted cache files...${NC}"
    
    for mesh_dir in "$TRAIN_MESH_DIR" "$VAL_MESH_DIR"; do
        for cache_type in "$CLUSTER_CACHE_DIR" "$CLUSTER_TEXTURE_CACHE_DIR"; do
            cache_path="$mesh_dir/$cache_type"
            if [ -d "$cache_path" ]; then
                echo "Checking $cache_type in $(basename "$mesh_dir")..."
                find "$cache_path" -name "*.pkl" -size 0 -delete 2>/dev/null || true
                echo -e "${GREEN}✓ Removed empty cache files from $cache_type${NC}"
            fi
        done
    done
}

clean_by_pattern() {
    local pattern="$1"
    echo -e "${YELLOW}Cleaning cache files matching pattern: $pattern${NC}"
    
    for mesh_dir in "$TRAIN_MESH_DIR" "$VAL_MESH_DIR"; do
        for cache_type in "$CLUSTER_CACHE_DIR" "$CLUSTER_TEXTURE_CACHE_DIR"; do
            cache_path="$mesh_dir/$cache_type"
            if [ -d "$cache_path" ]; then
                files_found=$(find "$cache_path" -name "*$pattern*" 2>/dev/null)
                if [ -n "$files_found" ]; then
                    echo "$files_found" | xargs rm -f
                    echo -e "${GREEN}✓ Removed files matching '$pattern' from $cache_type in $(basename "$mesh_dir")${NC}"
                else
                    echo "No files matching '$pattern' found in $cache_type in $(basename "$mesh_dir")"
                fi
            fi
        done
    done
}

# Main execution
case "${1:-info}" in
    "info"|"status")
        show_cache_info
        ;;
    "cluster")
        clean_cluster_cache
        ;;
    "texture")
        clean_texture_cache
        ;;
    "old")
        clean_old_cache
        ;;
    "corrupted")
        clean_corrupted_cache
        ;;
    "all")
        echo -e "${RED}WARNING: This will delete ALL mesh cache files!${NC}"
        echo "This will force recomputation of all mesh clusters and features."
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            clean_cluster_cache
            clean_texture_cache
            echo -e "${GREEN}All cache files cleaned!${NC}"
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    "pattern")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Pattern required for 'pattern' command${NC}"
            echo "Usage: $0 pattern <pattern>"
            echo "Example: $0 pattern 'clusters300'"
            exit 1
        fi
        clean_by_pattern "$2"
        ;;
    *)
        echo "Usage: $0 [info|cluster|texture|old|corrupted|all|pattern <pattern>]"
        echo ""
        echo "Commands:"
        echo "  info       - Show cache information and disk usage (default)"
        echo "  cluster    - Clean cluster cache (.cluster_cache directories)"
        echo "  texture    - Clean texture cache (.cluster_texture_cache directories)"
        echo "  old        - Clean cache files older than 7 days"
        echo "  corrupted  - Clean empty/corrupted cache files"
        echo "  all        - Clean ALL cache files (requires confirmation)"
        echo "  pattern    - Clean files matching a specific pattern"
        echo ""
        echo "Cache locations:"
        echo "  - $TRAIN_MESH_DIR/$CLUSTER_CACHE_DIR"
        echo "  - $TRAIN_MESH_DIR/$CLUSTER_TEXTURE_CACHE_DIR"
        echo "  - $VAL_MESH_DIR/$CLUSTER_CACHE_DIR"
        echo "  - $VAL_MESH_DIR/$CLUSTER_TEXTURE_CACHE_DIR"
        echo ""
        echo "Examples:"
        echo "  $0 info                    # Show cache status"
        echo "  $0 cluster                # Clean cluster cache only"
        echo "  $0 all                    # Clean everything"
        echo "  $0 pattern 'clusters300'  # Clean files with specific cluster count"
        exit 1
        ;;
esac

echo -e "${GREEN}Mesh cache cleaning completed!${NC}"
show_cache_info





