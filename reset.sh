#!/bin/bash

# Delete all files in the chapters folder
echo "🧹 Deleting all files in 'chapters/'..."
rm -f chapters/*

# Delete all .db files in the current directory
echo "🧨 Removing .db files..."
rm -f *.db

# Delete the audit_log.txt file
echo "🗑️ Deleting audit_log.txt..."
rm -f audit_log.txt

# Empty the structured_output.md file
echo "📄 Clearing contents of structured_output.md..."
: > structured_output.md

echo "✅ Reset complete."
