#!/bin/bash
BASE_DIR="$HOME/projects/project"

# delete raw files older than 7 days
find "$BASE_DIR/data/raw" -type f -mtime +7 -delete

# delete silver parquet older than 7 days
find "$BASE_DIR/data/lake/silver" -type f -mtime +7 -delete

# delete gold parquet older than 7 days
find "$BASE_DIR/data/lake/gold" -type f -mtime +7 -delete

# compress logs older than 1 day (rotate)
find "$BASE_DIR/logs" -type f -name "*.log" -mtime +1 -exec gzip {} \;

echo "[CLEANUP] Done $(date)" >> "$BASE_DIR/logs/cleanup.log"
