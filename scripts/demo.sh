docker run --gpus '"device=0"' --rm -v /home:/home -ti kglee/tf:2.5.0 python /home/Alexandrite/leekanggeun/CVPR/ISCL/scripts/demo_EM.py --iter=400 --epoch=20 --batch_size=64 --lr=1e-4 --kfold=4 \
--clean_data='/home/Alexandrite/leekanggeun/CVPR/data/EM/tem' \
--noisy_data='/home/Alexandrite/leekanggeun/CVPR/data/EM/simulated_image/tem_charge_noise' \
--result_dir='/home/Alexandrite/leekanggeun/CVPR/ISCL/result/'
