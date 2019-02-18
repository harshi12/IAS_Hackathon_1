FROM python


RUN apt-get update && apt-get install -y python3
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install tensorflow
RUN pip3 install matplotlib
RUN pip3 install sklearn


#create a working directory
RUN mkdir ./hackathon1


#copy all the required files 
COPY ./restore_model.py ./hackathon1
COPY ./train.csv 	./hackathon1
#COPY ./test.csv 	./hackathon1
COPY ./evaluation_submission.csv 	./hackathon1
COPY ./gender_submission.csv 	./hackathon1
COPY ./titanic.ckpt.meta ./hackathon1
COPY ./titanic.ckpt.index ./hackathon1
COPY ./titanic.ckpt.data-00000-of-00001 ./hackathon1

WORKDIR ./hackathon1

RUN chmod 777 /hackathon1/restore_model.py  
#RUN python3 /hackathon1/restore_model.py
