
The entire code can be found as a Jupyter Notebook. This can be open in two ways.

1) open Jupyternotebook. copy paste the url in local browse

2) we can launch using AWS
Procedure to RDP onto EC2 linux instance
## Follow each command step by step to create a user login for xrdp on EC2 linux boxes
sudo apt update &&  sudo apt upgrade

sudo sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config

sudo /etc/init.d/ssh restart

sudo passwd ubuntu

sudo apt install xrdp xfce4 xfce4-goodies tightvncserver

echo xfce4-session> /home/ubuntu/.xsession  


sudo cp /home/ubuntu/.xsession /etc/skel

sudo sed -i '0,/-1/s//ask-1/' /etc/xrdp/xrdp.ini

sudo service xrdp restart

sudo reboot

#Once the box reboots, go to putty to your windows remote desktop and type 3.249.182.152:IP  (Your instance IPv4 public IP:----) and hit connect (Username: ubuntu, password: The one set above)

## You can create a login for Windows server using the AWS portal by clicking Connect to Instance and Get password. 


## Using SSH to connect to jupyter notebook on EC2 ubuntu
## Install https://github.com/bmatzelle/gow/releases/download/v0.8.0/Gow-0.8.0.exe for running linux commands on windows

# Connect using ubuntu password
ssh ubuntu@host id (eg: mine-34.240.232.226)
Password: we set the password

# Connect using .pem key pair file
ssh -i new.pem ubuntu@host id

# Connect and bind local port 8000 to EC2's port 8888
ssh -i new.pem -L 8000:localhost:8888 ubuntu@host id

copy and paste the url in local browser.


mergedjsondata.csv - transformation of the data from JSON files for each month to a single .CSV file
ldavis_prepared_5.html - pyLDAvis- visualization output.

topicModeling_retailNews_Amani.ipynb - jupyter notebook
