#!/bin/bash
cd ..
mkdir datasettt
cd datasettt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KRqvea32EKuVym2H1nTyK-5MuSx4Y-P4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KRqvea32EKuVym2H1nTyK-5MuSx4Y-P4" -O cobre.zip && rm -rf /tmp/cookies.txt
unzip cobre.zip
rm cobre.zip