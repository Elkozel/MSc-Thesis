FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Set the workdir
WORKDIR /usr/src/noether

# Copy the project
COPY noether .

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Run the model
ARG model=Try2
ARG dataset=UWF22
ARG max_epochs=max_epochs

CMD [ "python run.py --model=$model --dataset=$dataset --max_epochs=$max_epochs" ]
