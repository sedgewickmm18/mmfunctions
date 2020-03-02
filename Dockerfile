FROM registry.access.redhat.com/ubi8:latest
RUN yum -y install cmake git pkg-config zip python36 make gcc gcc-c++ gcc-gfortran
RUN curl -L http://github.com/xianyi/OpenBLAS/archive/v0.3.7.zip > openblas.zip && unzip openblas.zip
RUN curl -L https://github.com/opencv/opencv/archive/4.2.0.zip > opencv.zip && unzip opencv.zip
RUN pip3 install numpy
RUN cd OpenBLAS-0.3.7 && cmake && make && make install
RUN cd opencv-4.2.0 && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..

