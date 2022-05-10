FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
LABEL maintainers="Amund Faller Raheim raheim@informatik.uni-freiburg.de"

# Copy from build context to image
COPY requirements.txt .

# Install requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir el-bert
WORKDIR /el-bert
COPY . .

# Change default Makefile
RUN mv Makefile Makefile_docker
RUN mv Makefile_scripts Makefile
# Append to .bashrc
RUN cat bashrc >> ~/.bashrc

# Build image with
# docker build -t el-bert .

# Run container with
# docker run -v /nfs/students/amund-faller-raheim/ma_thesis_el_bert/ex_data:/ex_data -v /nfs/students/amund-faller-raheim/ma_thesis_el_bert/data:/data -v /nfs/students/amund-faller-raheim/ma_thesis_el_bert/models:/models -it --name el-bert el-bert
