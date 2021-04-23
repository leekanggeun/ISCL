docker run --gpus '"device=0"' --rm -v /home:/home -ti ISCL:2.5.0 python demo_EM.py --iter=400 --epoch=20 --batch_size=64 --lr=1e-4 --kfold=4 \
--clean_data='' \
--noisy_data='' \
--result_dir=''
