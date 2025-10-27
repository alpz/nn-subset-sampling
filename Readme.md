This code accompanies the ICLR 2023 paper "Scalable Subset Sampling with Neural Conditional Poisson Networks", Adeel Pervez, Philip Lippe, Efstratios Gavves.

The repository contains code for running the model explanation experiments for text and image classification models on 20Newsgroups and STL-10.

There is a .yml file include to create the required conda environment.

----------------------------
Model Explanation for STL-10.
----------------------------

Model to be explained:
The model to be explained in this case achieved about 75% evaluation accuracy. The test and train predictions for the model are stored in the out/stl10 directory.

Pre-trained explainer model:
A pretrained explainer for the STL-10 model above is stored in the 'out/l2x_stl10/learning_rate_0.0001.model_L2XSTL10.t_1.task_stl10' directory.
This model generates k=700 size k-hot masks which are applied to images and the subsequent classifier attempts to reproduce the predictions of the model to be explained using the masked image.

To run the pretrained model activate the conda environment and run
$ python l2x_train_stl10.py --resume --eval --t 1

This downloads the dataset to ~/datasets if not available.
This should output a post hoc accuracy of about 60%. This should also save some masked images in the directory with the model.

To train a new explainer from scratch run
$ python l2x_train_stl10.py 


You can also change the subset size and whether the subset size is differentiable by using the subset_size and the diffk flags. You can change the regularization weight (currently 0.01) for diffk in the model definition in model/l2x_stl10.py

To train a new classifier to explain you can run the script in stl10_classifier.py. It should store the predictions in out/stl10 overwriting any existing ones.

To train the Gumbel Top-k baseline from https://github.com/ermongroup/subsets use the --model L2XSTL10Subop argument. Note that this might cause an OOM error with the default subset size (700) and batch size (128). You would need to reduce the subset size to about 100 and batch size to about 32 to make the baseline work.


--------------------------------------------------------
Model Explanation for text classification on 20Newsgroups:
---------------------------------------------------------

In this case the test data and test predictions of the model to be explained are stored in out/20ng

Run pre-trained model on 20newsgroups with differentiable set size with average size 50 on the test data and predictions stored in out/20ng.
python train_l2x_text.py --diffk --resume --eval --t 1 --correct
This should give a post hoc accuracy of over 68% and save some samples in the experiment output directory. The accuracy without the correction is slightly lower.

You can train a new explainer for the predictions stored in out/20ng by running
python train_l2x_text.py
You can set the subset_size and diffk to change the subset size and whether k is differentiable and whether to apply the inclusion probability correction.

To train a new model to be explained download the 20newsgroups data to the "~/datasets" directory and GLoVe embeddings to the embeddings directory.
Download GLoVe embeddings 'glove.6B.100d.txt'from https://nlp.stanford.edu/projects/glove/ and store them in the 'embeddings' directory.
Then run the train_l2x_text.py script to train the model to be explained which should overwrite the predictions in the out/20ng directory.


To run the RelaxSubSample baseline on 20Newsgroups use the --model L2X20ngSubop argument.


