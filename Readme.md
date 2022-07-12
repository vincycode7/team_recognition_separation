To run solution locally

Step 1. install all requirements in requirements.txt
step 2. run python main.py

To view result of the video analysis toggle the display_result_to_gui to True in the main.py file before running the program
# Result will be saved in output folder

To run docker solution
Step 1. install docker locally
Step 2. Run `docker build -t docker-parallelscore-football-seperator-model -f Dockerfile .`
Step 3. When build is complete Run `docker run docker-parallelscore-football-seperator-model python main.py` 

Enjoy !!!!!
