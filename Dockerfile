FROM tensorflow/tensorflow:2.4.0

# Set working directory
WORKDIR /root/workdir

# Install dependencies
RUN apt-get update
RUN apt-get install -y build-essential git time
RUN git clone --branch develop --recurse-submodules https://github.com/AlessioZanga/REILLY.git
RUN cd REILLY && pip install -r requirements.txt
RUN cd REILLY && python setup.py develop

# Copy files to container
COPY demo.py .

# Run command
CMD ["/usr/bin/time", "-v", "python", "./demo.py"]
