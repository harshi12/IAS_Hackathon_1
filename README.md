# IAS_Hackathon_1

git link: https://github.com/harshi12/IAS_Hackathon_1

----- SSH Approach-----
On server side, in file run.sh, configure the ip, username, password of the client machine on which you want to test the deployed model.
Client side will receive the file related to the trained model and an output file, named output.txt, will be generated at run time using the test data and the given model. Output will be displayed on server portal as well.

-------Docker Approach--------------
Client will be provided with the Dockerfile which will have configurations of the desired docker image. bash script, run_docker.sh will build and run the docker image that will print the output on client terminal.
