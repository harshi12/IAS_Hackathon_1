pass="@14799741hA"
pass1="rama1729"
username="harshita"
ip="192.168.43.174"
# # echo $pass |ssh harshita@192.168.43.174

sshpass -p "$pass" scp script.sh $username@$ip:temp/script.sh
sshpass -p "$pass" scp restore_model.py $username@$ip:temp/restore_model.py
sshpass -p "$pass" scp checkpoint $username@$ip:temp/checkpoint
sshpass -p "$pass" scp titanic_graph.meta $username@$ip:temp/titanic_graph.meta
sshpass -p "$pass" scp titanic.ckpt.data-00000-of-00001 $username@$ip:temp/titanic.ckpt.data-00000-of-00001
sshpass -p "$pass" scp titanic.ckpt.meta $username@$ip:temp/titanic.ckpt.meta
sshpass -p "$pass" scp titanic.ckpt.index $username@$ip:temp/titanic.ckpt.index
sshpass -p "$pass" scp gender_submission.csv $username@$ip:temp/gender_submission.csv
sshpass -p "$pass" scp test.csv $username@$ip:temp/test.csv
sshpass -p "$pass" ssh $username@$ip << EOF
cd temp
bash script.sh
# sshpass -p "rama1729" scp output.txt kaushik@192.168.43.189:Desktop/output.txt
EOF
# sshpass -p "$pass" ssh harshita@192.168.43.174