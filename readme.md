```
pip install jupytext
jupytext --set-formats ipynb,py:percent "CAB320-Assignment2-Learning.ipynb"
jupytext --to notebook "CAB320-Assignment2-Learning.py"
jupytext --sync "CAB320-Assignment2-Learning.ipynb"
```