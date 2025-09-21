#!/bin/bash

docker build --pull --rm -f 'Dockerfile' -t 'simplenn:latest' '.' 
docker run --rm -it -v ./graphs:/app/graphs simplenn:latest