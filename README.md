<h1 style="text-align: center">Netfloox</h1>
<p style="text-align: center">Recommendation and popularity prediction</p>

## Setting up the environment
The environment is made using python 3.11, you first need to make sure it is installed on your system :
### Python 3.11 installation :
Ubuntu/Mint/Debian :
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install python3-pip
```
Arch/Manjaro/Endeavouros :
```
sudo pacman -Sy python
sudo pacman -S python-pip
```
MacOS (with homebrew https://brew.sh/) :
```
brew install python@3.11
```
### mariadb installation :
Ubuntu/Mint/debian :
```
sudo apt update
sudo apt install mariadb-server
```
Arch/Manjaro/Endeavouros :
```
sudo pacman -Sy mariadb
```
MacOS (with homebrew https://brew.sh/) :
```
brew install mariadb
```
### Creation of the environment for linux/macOS :
Open a terminal and navigate into the repositoriy directory with the cd command and launch the script with :
```
sh create_env_linux.sh
```
To activate the env use :
```
source env/bin/activate
```
while in the repository folder.
