#!/bin/bash
# Start database server for retool-summary system

set -e

cd "$(dirname "$0")"

echo "Starting database server..."
python database_server.py

