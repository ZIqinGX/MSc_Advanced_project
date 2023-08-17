## To make sure this program run correctly, first need to download and install several important library in environment and here are record of how I set right enviornment and what problem I met.

1. Create new environment<br/>
```
 conda create --name fyp1 python=3.11
 conda activate fyp1
 python -m pip install tensorflow
 conda install -c conda-forge -y pandas jupyter
 python -m ipykernel install --user --name fyp1
```
Now I have created a virtual environment called **fyp1**

2. Download library called **music21**<br/>
Reference from https://web.mit.edu/music21/doc/installing/installWindows.html <br/>
**Note**:Windows users should download and install Python version 3.10 or higher. That's why I set python=3.11 before.
```
pip install music21
```
then
```
python3 -m music21.configure
```
3. Download **Scikit-Learn library**<br/>
```
conda install scikit-learn
```
4.Install **seaborn**<br/>
```
conda install seaborn
```
5.Install **pyfluidsynth** module
```
pip install pyfluidsynth
```
6.Install **pretty_midi**
```
pip install pretty_midi
```
7.Download**midi2audio**
```
pip install midi2audio
```



