#!/bin/bash
for i in $(seq 0.1 0.1 1)
do
   ./ytrain -s 15 -L $i ../../Data_ML/webspam.train _model
   ./ypredict ../../Data_ML/webspam.test _model _out
done
for j in $(seq 1.5 0.5 50)
do
   ./ytrain -s 15 -L $j ../../Data_ML/webspam.train _model
   ./ypredict ../../Data_ML/webspam.test _model _out
done