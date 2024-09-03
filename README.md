This is the code base for the SEDE-FILM-method presented in
the bachelor thesis written by Leon Forth. Many ideas and some 
code snippets originate from the code in https://github.com/valerfisch/geo-scene.

After downloading the datasets (so running all the
files in the preconfigs folder) and run the file get_semantic_masks
once to get the STEGO masks 
you should be able to run the method (main.py). 

To change the settings (the clustering method and the parameters
or if filtering classes oder the FILME-method should be applied)
please look into the file configs/eval_config.

The datasets can be found in the datasets folder, all the
helper-files in the utils folder.

For setting up a mamba-environment please have a look at the requirements.txt file.
There you can find all the packages I used in my set-up.
