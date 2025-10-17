#!/bin/bash
# Kill all benchmark processes (but NOT Ollama servers)

echo "Killing all benchmark processes..."

# Kill Python processes running livebench benchmarks
pkill -f "python.*run_livebench.py" 2>/dev/null && echo "  ✓ Killed Python benchmark processes" || echo "  ℹ No Python benchmark processes found"

# Kill any remaining processes connected to Ollama ports (11434-11440), excluding ollama itself
for port in {11434..11440}; do
    # Get PIDs connected to this port
    PIDS=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$PIDS" ]; then
        for pid in $PIDS; do
            # Skip if it's an ollama process
            if ! ps -p $pid -o command= 2>/dev/null | grep -q "ollama"; then
                kill -9 $pid 2>/dev/null && echo "  ✓ Killed process $pid on port $port"
            fi
        done
    fi
done

echo "Done!"
