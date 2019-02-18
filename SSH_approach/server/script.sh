# pass="@14799741hA"
# echo $pass | sudo -S apt-get update
# echo $pass | sudo apt-get install python
# echo $pass | sudo apt-get pip
# echo $pass | sudo pip install numpy
# echo $pass | sudo pip install pandas
# echo $pass | sudo pip install tensorflow
python3 restore_model.py
sshpass -p "rama1729" scp ~/temp/output.txt kaushik@192.168.43.189:output.txt