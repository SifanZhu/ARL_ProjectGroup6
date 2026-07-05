GENERAL SETUP STEPS FOR PYTHON ENVIRONMENT:
============================================

1) Install PyENV: https://github.com/pyenv/pyenv

2) Download / Make sure that PyEnv has instance of Python 3.11.9 environment
  => pyenv update
  => pyenv install 3.11.9

3) Create local directory for the lecture exercise and CD into it, e.g. c:\exercise\model_based_rl
  => cd c:\exercise\model_based_rl
  
4) Copy lecture files (requirements.txt and Python-Notebook) into it

5) Create instance of Python 3.11.9
  5.1) creates .python-version file containing the desired version number
    => pyenv local 3.11.9

  5.2) find out path to python 3.11.9
    => pyenv which python
    => gives e.g.: C:\Users\<USER>\.pyenv\pyenv-win\versions\3.11.9\python.exe

  5.3) create the virtual Python environment with correct python version
    => C:\Users\<USER>\.pyenv\pyenv-win\versions\3.11.9\python.exe -m venv .venv

6) Activate created virtual env  
  => (on Windows) .venv\scripts\activate
  => (on Linux)   source .venv/bin/activate

7) Install required packages from requirements.txt into .venv
  => pip install -r requirements.txt


=====================================================================================

WORK WITH THE PYTHON NOTEBOOK: 
==============================

From now on you can use the created Python environment for working on the exercise ipynb. 

1) Make sure Python environment is activated 
  => (on Windows) .venv\scripts\activate
  => (on Linux)   source .venv/bin/activate

2) Start Python-Notebook
  => jupyter notebook
