# Objectives of the Project

Detect in approximately which frame the door opens or closes.

# Method

Calculate optical flow, define the strength of optical flow through weighted magnitude and angle, and filter out noise and moving crowd lines using masks based on color and edge features to retain primarily straight lines corresponding to bus doors. Use find_peak to identify more prominent peaks in optical flow, defined as the frames when events (door opening or closing) occur, and ensure that the same event is not mistakenly interpreted as two separate events by merging neighboring peaks.

# Environment Setup

> conda create -n CV_Final python==3.11.4
> conda activate CV_Final
> cd /path/to/your/final_project
> pip install -r requirements.txt


# Execute the program

Move to the final_project folder.

	cd /path/to/your/final_project

Adjust the following code in the main function of algorithm.py to the folder containing the videos to be tested:

	directory = "Tests" # Specify the directory to scan

Then execute the following code:
	
	python algorithm.py

The output.json will appear in the final_project folder.