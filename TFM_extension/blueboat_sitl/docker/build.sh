#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

docker build -f $DIR/Dockerfile -t blueboat_sitl:latest ..

#if you need build from scratch 
# docker build --no-cache -f $DIR/Dockerfile -t blueboat_sitl:latest ..
