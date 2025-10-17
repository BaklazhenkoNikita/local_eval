#!/bin/bash
# Ollama Manager: ./start_ollama.sh [action] [model]

ACTION=${1:-serve}
MODEL=${2:-qwen2.5:3b}

case $ACTION in
    serve|start)
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && { echo "Already running"; exit 0; }
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 2
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "Started" || echo "Failed"
        ;;
    
    pull)
        ollama pull "$MODEL"
        ;;
    
    list|ls)
        ollama list
        ;;
    
    stop)
        pkill -f "ollama serve" && echo "Stopped" || echo "Not running"
        ;;
    
    status)
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && ollama list || echo "Not running"
        ;;
    
    *)
        echo "Usage: $0 [serve|pull|list|stop|status] [model]"
        exit 1
        ;;
esac
