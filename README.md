# learn2dance

<https://github.com/ardapekis/learn2dance>

*insert how to use code here*

# Using the Discriminative Model

## Processing the Data
1. Run `extract_poses.sh`. This will take a while, and requires GPUs and nvidia-docker.
2. Run `Clustering.ipynb`. This will run kmeans on the poses and save the centroids.
3. Run `Process_Waves.ipynb`. This will save music features, extracted with librosa.

## Training and Evaluation
For training:
Run `train.py` or use `Model.ipynb`. This will train and save the model.
For evaluation:
Run `eval.py` or use `Model.ipynb`. This will save an animation of the predicted dance.

# Advice for Future Work

The most obvious task would be to get a higher accuracy with the results, and collecting and training on more data would definitely help with that. In addition, it may be a good idea to focus on one specific type of dance or song category, and attempt to collect data that specifically relates to that one type of dance, such as hip-hop, and get good results on that before including other categories. We may have been a bit ambitious with this project and better results may be obtained by training the model category-wise. 

But above all - have fun with this! We enjoyed working on this project and it was a lot of fun, regardless of the results. We all learned a lot and who knows? Maybe you'll even learn2dance a little :D
