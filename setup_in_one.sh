# !/bin/bash
set -e
cd ~
#sudo apt update && sudo apt install zsh
#sudo wget -nv -O - https://raw.githubusercontent.com/zimfw/install/master/install.zsh | zsh
#cp /mnt/disks/storage/Anaconda3-2024.02-1-Linux-x86_64.sh ./
#./Anaconda3-2024.02-1-Linux-x86_64.sh
#conda init zsh
#conda create -n xla_py38 python==3.8 && conda activate xla_py38
#pip install torch~=2.2.0 torch_xla\[tpu\]~=2.2.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html

# sudo apt-get install ffmpeg libsm6 libxext6 # required for opencv (cv2)

# never do anything about ipynb! this will ruin your whole conda env

#git config --global user.name bytetriperTPU && git config --global user.email bytetriper@TPU.google.com &&ssh-keygen -t rsa -C bytetriper@gmail.com
#ssh -T git@github.com

HOME_PATH=/home/bytetriper

# create soft link
# define a list of files to link
# we'll need a sudo to link the .bash_history
files=(.zshrc .bashrc .bash_logout .profile)
storage_path=/mnt/disks/storage/config
echo "storage path: ${storage_path}"
sudo mkdir ${storage_path} -p
sudo chmod -R 777 ${storage_path}
# filter files that do not exist
files=($(for file in ${files[@]}; do if [ -e $HOME_PATH/$file ]; then echo $file; i; done) )
# filter the file that exists
#files=($(for file in ${files[@]}; do if [ ! -e $storage_path/$file ]; then echo #$file; fi; done) )
# filter the file that already is a link
files=($(for file in ${files[@]}; do if [ ! -L $HOME_PATH/$file ]; then echo $file; fi; done) )
# if files is empty, then exit
if [ ${#files[@]} -eq 0 ]; then
    echo "no files to link"
    ls -l -a $HOME_PATH
    exit 0
fi
# echo files
echo "files to link:"
echo ${files[@]}
# ask for confirmation
read -p "Do you want to link these files? [y/n] " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ##first mv them to storage
    for file in ${files[@]}; do
        sudo mv $HOME_PATH/$file $storage_path
    done
    ## then chmod them to +x
    for file in ${files[@]}; do
        sudo chmod -R +x $storage_path/$file
    done
    ## then link them
    for file in ${files[@]}; do
        sudo ln -s $storage_path/$file $HOME_PATH/$file
    done
fi
echo "\ndone"
ls -l -a $HOME_PATH