#!/bin/bash

# Script to create and manage SGLang Docker container

CONTAINER_NAME="llm42"
IMAGE_NAME="lmsysorg/sglang:v0.5.4"

# Function to display usage
usage() {
    echo "Usage: $0 [create|attach|stop|restart|status]"
    echo ""
    echo "Commands:"
    echo "  create   - Pull image and create container"
    echo "  attach   - Attach to running container"
    echo "  stop     - Stop the container"
    echo "  restart  - Restart the container"
    echo "  status   - Check container status"
    echo ""
    echo "If no command is provided, 'create' will be executed"
}

# Function to create container
create_container() {
    echo "Pulling Docker image: $IMAGE_NAME"
    docker pull $IMAGE_NAME
    
    echo ""
    echo "Creating container: $CONTAINER_NAME"
    docker run -d \
        --name $CONTAINER_NAME \
        --gpus all \
        --ipc=host \
        --network host \
        -v $(pwd):/workspace \
        $IMAGE_NAME \
        sleep infinity
    
    if [ $? -eq 0 ]; then
        echo "Container '$CONTAINER_NAME' created successfully!"
        echo "To attach to the container, run: $0 attach"
    else
        echo "Failed to create container"
        exit 1
    fi
}

# Function to attach to container
attach_container() {
    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Attaching to container: $CONTAINER_NAME"
        docker exec -it $CONTAINER_NAME /bin/bash
    elif docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container exists but is not running. Starting it first..."
        docker start $CONTAINER_NAME
        sleep 2
        docker exec -it $CONTAINER_NAME /bin/bash
    else
        echo "Container '$CONTAINER_NAME' does not exist. Run '$0 create' first."
        exit 1
    fi
}

# Function to stop container
stop_container() {
    echo "Stopping container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
}

# Function to restart container
restart_container() {
    echo "Restarting container: $CONTAINER_NAME"
    docker restart $CONTAINER_NAME
}

# Function to check status
check_status() {
    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' is RUNNING"
        docker ps --filter "name=$CONTAINER_NAME"
    elif docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' exists but is NOT RUNNING"
        docker ps -a --filter "name=$CONTAINER_NAME"
    else
        echo "Container '$CONTAINER_NAME' does NOT EXIST"
    fi
}

# Main script logic
case "${1:-create}" in
    create)
        create_container
        ;;
    attach)
        attach_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    status)
        check_status
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
