### What?

Python code created while reading through the book:

Hands-On Large Language Models  
by Jay Alammar, Maarten Grootendorst  
Released September 2024  
Publisher(s): O'Reilly Media, Inc.  
ISBN: 9781098150969  

https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/

The official source code can be found at:  
https://github.com/HandsOnLLM/Hands-On-Large-Language-Models

### Why?
You learn more if you work out the code & mess with it rather than relying on other people's code - so this matches up w/ what the book does, but it also contains some other little bits and pieces of experimentation and nicities like caching any data files we need rather than re-downloading them each run.  

Also, there's a ton of notes & comments alongside the code to help me (and you?) understand exactly what the hell's going on! 

Many thanks to Jay Alammar & Maarten Grootendorst for their terrific book =D

### Python Environment Notes

Rather than clutter your system's Python libraries it's likely best to use a separate Python environment. I quite like [PyEnv](https://github.com/pyenv/pyenv).

To use PyEnv, install it as you see fit (e.g., from a GitHub release, or it's `sudo pacman -Suy pyenv` on Arch) then go:

`pyenv install --list` - to list all python versions which CAN be installed  
`pyenv install <some-version>` - to install a given python version  
`pyenv versions` - to list all python versions which ARE installed and can be used  
`pyenv local <some-version>` - INSIDE THE DIRECTORY you want to use that version!

I'm using Python 3.11.13 for this repo, so if I was doing a fresh setup I'd clone this repo then into it and go:  
`pyenv install 3.11.13` (if 3.11.13 wasn't already installed as an available environment)  
`pyenv local 3.11.13` (to set the repo to use the specific version), then  
`python --version` (to double-check that it's actually pointing to 3.11.13 not the system Python interpreter).

**Note**: [PyCharm](https://www.jetbrains.com/pycharm/) should automatically use the correct `.venv` folder when you've set a local Python environment, but if you're using a different IDE you may need to go `source .venv/bin/activate` - see the PyEnv docs for further details. 
