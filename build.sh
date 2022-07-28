#! /bin/bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.12.0-linux-x86_64.tar.gz ~
tar xvzf ~/elasticsearch-7.12.0-linux-x86_64.tar.gz 

# pypi
pip install -r requirements.txt

# prepare data
wget --content-disposition https://cloud.tsinghua.edu.cn/f/8c01c331523b43088bac/?dl=1 
unzip data.zip
rm data.zip 




