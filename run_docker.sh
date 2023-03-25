
#!/bin/sh
# replace with your root folder absolute path
ROOT_PATH=$(realpath $(dirname $0))
echo "Test project path: $ROOT_PATH"
# deal with the permision problem
docker build -t juntao/lptorch:0.1 .
CONT_NAME="lptorch" 
# run if the docker container is not running but has been created
if [ "$(docker ps -aq -f name=$CONT_NAME)" ]; then
    echo "Container $CONT_NAME is running"
    docker start $CONT_NAME
    exit 0
fi
docker run --net=host --gpus all --name $CONT_NAME -v $ROOT_PATH:/workspace -it juntao/lptorch:0.1 

