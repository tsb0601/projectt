export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update 
#sleep for 10 seconds
sleep 15
sudo killall dpkg apt apt-get
sudo dpkg --configure -a # dpkg may get corrupted
sleep 10
sudo apt-get install gcsfuse
#
mkdir ckpt_gcs -p
gcsfuse --config-file gcsfuse_config tsb-data-1 "ckpt_gcs"

# fusermount -u VAE-enhanced/ckpt_gcs
