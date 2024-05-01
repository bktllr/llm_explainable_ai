# Introduction
This is the code that was used for the use-case example presented in the paper "XXX".

# Setup
The following document describes the initial setup steps.

## Text-generation-webui
The Text-generation-webui-folder is based on the text-generation-webui developed by Oobabooga: [https://github.com/oobabooga/text-generation-webui]. The following parts describe the initial setup of a conda-environment for the text-generation-webui, the download of the text-generation-webui and how the text-generation-webui can be started.

### Setup of Conda environment

The following commands have been executed within the terminal of jupyter-lab to setup a conda-environment for the text-generation-webui:

    conda create -n textgen python=3.11.6 -y
    conda activate textgen
    
### Extra kernel for this conda environment

To also use this conda-environment in a jupyter-notebook, a specific kernel was created based on the environment usind the following terminal code:

    (base) conda install nb_conda_kernels
    (base) conda activate textgen
    (textgen) conda install -c anaconda ipykernel
    (textgen) python -m ipykernel install --user --name=textgen
    
    
### Download text-generation-webui
The following commands have been executed within the terminal of jupyter-lab to download and setup the text-generation-webui:

    python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    git clone https://github.com/oobabooga/text-generation-webui
    cd text-generation-webui
    python3.11 -m pip install -r requirements.txt
    python3.11 -m pip install intel_extension_for_pytorch
    
### Starting the webui
The text-generation-webui can be startet via the terminal using the following commands:

    conda activate textgen
    cd /home/[...]/text-generation-webui/
    python server.py --share --model models/Llama-2-13b-chat-hf

The textgeneration-webui runs on a local and public url. Copy the public url into your browser to access the webui.

## Gradio-webui
To initialize the artefact in a chatbot interface, we firstly setup a new conda-environment. Therefore, we used the following commands in the terminal:

    conda create -n webui python=3.10.9 -y
    conda activate webui
    python3.10 -m pip install transformers gradio torch accelerate catboost shap

Afterwards, we created a gradio-app [https://www.gradio.app/] in the file "Webui.py" in the folder "Gradio-webui". The gradio-app can be started using the following commands in the terminal:

    conda activate webui
    cd /home/[...]/Gradio-webui/
    python Webui.py

Attention: If you want to use this gradio app, you have to enter your huggingface-token in line 6 in the document "webui.py".

The gradio app runs on a local and public url. Copy the public url into your browser to access the webui.

## Copyright
We acknowledge that a significant portion of the code in this GitHub repository is derived from research conducted by a research team around Alina Buss at the Technical University of Munich. The work of Alina Buss and her research team has served as the foundation for the development of this software and formed the basis for the corresponding paper. 
