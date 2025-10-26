#!/bin/bash

# Quick Chrome installer for etalon/kaleido
# This script installs Chrome for plot generation in etalon benchmarks

echo "================================================"
echo "Installing Chrome for Etalon/Kaleido"
echo "================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Method 1: Try using kaleido's built-in installer
echo "Attempting to install Chrome using kaleido..."
if python3 -c "import kaleido; kaleido.get_chrome_sync()"; then
    echo "✓ Chrome installed successfully via kaleido"
    exit 0
fi

# Method 2: Try using plotly's installer
echo ""
echo "Attempting to install Chrome using plotly..."
if command -v plotly_get_chrome &> /dev/null; then
    if plotly_get_chrome; then
        echo "✓ Chrome installed successfully via plotly_get_chrome"
        exit 0
    fi
fi

# Method 3: Try using kaleido command
echo ""
echo "Attempting to install Chrome using kaleido command..."
if command -v kaleido_get_chrome &> /dev/null; then
    if kaleido_get_chrome; then
        echo "✓ Chrome installed successfully via kaleido_get_chrome"
        exit 0
    fi
fi

# If all methods fail, provide manual instructions
echo ""
echo "================================================"
echo "Automatic installation failed"
echo "================================================"
echo ""
echo "Please install Chrome manually using one of these methods:"
echo ""
echo "Option 1 - Python command:"
echo "  python3 -c 'import kaleido; kaleido.get_chrome_sync()'"
echo ""
echo "Option 2 - Command line tool:"
echo "  plotly_get_chrome"
echo "  or"
echo "  kaleido_get_chrome"
echo ""
echo "Option 3 - Install Chrome manually:"
echo "  Follow Google's instructions for your operating system"
echo "  https://www.google.com/chrome/"
echo ""
exit 1
