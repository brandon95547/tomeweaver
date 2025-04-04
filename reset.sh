#!/bin/bash

# Delete all files in the toc folder
echo "🧹 Deleting all files in 'toc/'..."
find toc/ -type f ! -name 'full.md' -delete

# Delete all .db files in the current directory
echo "🧨 Removing .db files..."
rm -f *.db

# Delete the audit_log.txt file
echo "🗑️ Deleting audit_log.txt..."
rm -f audit_log.txt

echo "✅ Reset complete."
