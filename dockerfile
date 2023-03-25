FROM springtonyzhao/torch-lowprecision:latest
ARG PROJECT_DIR=/workspace
ADD . $PROJECT_DIR
WORKDIR $PROJECT_DIR
# change data path and number of trainers here
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 ca-certificates curl jq wget \
    git-lfs


CMD echo "===========END========="
CMD /bin/bash