# BUILD WITH:   "sudo docker build -t my-costumized-tensorflow-1-15 ."
# RUN WITH:     "sudo docker run -u $(id -u):$(id -g) -it --rm -v ~/openAI_testing/Reinforcement_Learning:/my_repository --runtime=nvidia my-costumized-tensorflow-1-15:latest bash"
# GPU CKECK:    "nvidia-smi"

# define base image
FROM tensorflow/tensorflow:1.15.2-gpu-py3

# create repo direcotry
RUN mkdir -p "my_repository"

# copy dependencies
COPY ./requirements-py.txt "my_repository"
COPY ./requirements-apt.txt "my_repository"

# install dependencies
RUN pip3 install -r /my_repository/requirements-py.txt
# WTF does this not work????
# RUN apt-get install -y requirements-apt.txt 
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        nano \
        python-opengl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /
ENV OPENCV_VERSION="4.1.0"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.6 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.6) \
  -DPYTHON_INCLUDE_DIR=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.6 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}
