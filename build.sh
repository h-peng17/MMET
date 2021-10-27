#! /bin/bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.12.0-linux-x86_64.tar.gz ~
tar xvzf ~/elasticsearch-7.12.0-linux-x86_64.tar.gz 

pip3 install hnswlib --user 
pip3 install Pillow --user 
pip3 install matplotlib --user 
pip3 install pytorch_metric_learning --user 
pip3 install elasticsearch --user
pip3 install nltk --user


# prepare data
mkdir data



