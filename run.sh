VAR1=PYTHONPATH=/home/oscar/py-faster-rcnn-fp16/caffe-fast-rcnn/python 
VAR2=LD_LIBRARY_PATH=/usr/local/cuda/lib:caffe-fast-rcnn/.build_release/lib
ENV="$VAR1 $VAR2"

#DATA=../VOC2012/JPEGImages
DATA=data/images

sudo $ENV ./detect.py $DATA
#sudo $ENV ../power_tools/profile_app.py ./detect.py $DATA | tee out.std && ../power_tools/print_power.py power.txt out.std
#sudo $ENV ../power_tools/profile_app.py ./detect.py $DATA | tee out.std && ../power_tools/print_power.py power.txt out.std | tee log/layers$(date +%Y%m%d%H%M).log
