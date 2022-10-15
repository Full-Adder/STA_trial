# 训练d
# python train.py --STA_mode "S" --input_size 256 --batch_size 64 --epoch 30 --save_dir "/media/ubuntu/Data/Result/"
# python train.py --STA_mode "SA" --input_size 256 --batch_size 40 --epoch 30 --save_dir "/media/ubuntu/Data/Result/"
# python train.py --STA_mode "ST" --input_size 256 --batch_size 20 --epoch 30 --save_dir "/media/ubuntu/Data/Result/"
# python ./coarse2fine/generate_CAM.py --STA_mode "S"  --input_size 256 --batch_size 550 --epoch 30 --save_dir "/media/ubuntu/Data/Result/" --Att_re_path "/media/ubuntu/Data/Result/Att_30"
# python ./coarse2fine/generate_CAM.py --STA_mode "SA" --input_size 256 --batch_size 170 --epoch 30 --save_dir "/media/ubuntu/Data/Result/" --Att_re_path "/media/ubuntu/Data/Result/Att_30"
# python ./coarse2fine/generate_CAM.py --STA_mode "ST" --input_size 256 --batch_size 500 --epoch 30 --save_dir "/media/ubuntu/Data/Result/" --Att_re_path "/media/ubuntu/Data/Result/Att_30"
# python ./coarse2fine/see_mode_in_train.py --STA_mode "SA" --input_size 256 --batch_size 100 --save_dir "/media/ubuntu/Data/Result/"
# python ./coarse2fine/Coarse_refuse_SCAM.py --Att_re_path "/media/ubuntu/Data/Result/Att_30"
#python ./coarse2fine/CAMsmooth_and_crop.py --Att_re_path "/media/ubuntu/Data/Result/Att_30" --Crop_path "/media/ubuntu/Data/Result_crop/Att_30"
python train.py --STA_mode "S"  --input_size 356 --batch_size 10 --epoch 30 --Pic_path "/home/ubuntu/AVE_Dataset/Crop_Att_30/" --save_dir "/media/ubuntu/Data/Result_crop/"
python train.py --STA_mode "SA" --input_size 356 --batch_size 10 --epoch 30 --Pic_path "/home/ubuntu/AVE_Dataset/Crop_Att_30/" --save_dir "/media/ubuntu/Data/Result_crop/"
python train.py --STA_mode "ST" --input_size 356 --batch_size 10 --epoch 30 --Pic_path "/home/ubuntu/AVE_Dataset/Crop_Att_30/" --save_dir "/media/ubuntu/Data/Result_crop/"




