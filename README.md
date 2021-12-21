.
    Voice Separate Models for speech application

    ├── README.md
    
    ├── code: core model for speech separate
        ├── __init__.py
        ├── __pycache__
        ├── conf.yml
        ├── conv_tasnet.py
        ├── data.py
        ├── dprnn-train-deploy.py
        ├── inference.py
        ├── parser_utils.py
        ├── requirements.txt
        ├── wham_dataset_no_sf.py
        └── wsj0_mix.py

    ├── conf.yml: config file for model

    ├── helper: helper function for application
        ├── __pycache__
        └── utils.py

    ├── local_experiment.ipynb: local experiment

    ├── train_and_deploy.ipynb: implementation in Sagemaker

    ├── mix_both.json: config for mix application

    ├── result: sample results
        ├── complete_split_s1.wav
        └── complete_split_s2.wav

    ├── split_weight: best model for speech application
        └── best_model.pth
    
    ├── test_sample: test sample for speech application
        └── audio_raw_1-20_clip_0.wav
    

    └── train_sample_v6: sample for train/validation
        ├── cv
        ├── tr
        └── tt

11 directories, 20 files
