Environment Builds:

	1.Download openCV, Tensorflow, Keras to your computer
	2.Edit the paths in all documents 

Running Flow:

For new users:
	1. Open ui.py, enter your name in the textfield under 'Your Name', and click 
	'Capture' to get 50 pictures. (If you want more, you can change the parameter in 
	getFace.py) Then replace the name to another new user, do the same thing.
	2. After all new user finished capture, close ui.py, run CNN_train.py, enter your 	names and it will create models for new users.
	3. Open ui.py again, enter the name of two users that you want to recognize 	
	together, and enter the emotion that you want to be used for emotion analysis. 	
	then click 'Recognize', and it starts to recognize for a few seconds, and show the 	results

For old users:
	Run step 3 directly.

Emotion model:

	You don't need to run emotion.py to get emotion model. It is already in the zip 
	file. However, if you want to try, run emotion.py and will take several minutes to 
	train, and create a model for emotion.
