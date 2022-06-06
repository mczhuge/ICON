#!/usr/bin/env python3
# coding=utf-8
import os
dir = './GT'
txt = './train.txt'
f = open(txt,'a')
for filename in os.listdir(dir):
    f.write(filename.split('.')[0])
    f.write("\n")
f.close()
