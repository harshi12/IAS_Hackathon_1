#!/bin/bash
echo "Enter password of root user"
read password

echo $password | sudo -S docker build -f Dockerfile -t  mydocker .
sudo docker run -it --name mydock4 mydocker
sudo docker cp ./test.csv mydock4:/hackathon1/test.csv
#sudo docker exec -it mydock3 execfile("restore_model.py")
sudo docker stop mydock4
sudo docker remove mydock4
