FROM nvcr.io/nvidia/pyg:25.05-py3

# Set the workdir
WORKDIR /usr/src/noether

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the project
COPY noether .

# Run the model
ARG model=Try2
ARG dataset=UWF22
ARG max_epochs=max_epochs

CMD [ "tail", "-f","/dev/null" ]
# CMD [ "python", "run.py", "--model=$model", "--dataset=$dataset", "--max_epochs=$max_epochs" ]
