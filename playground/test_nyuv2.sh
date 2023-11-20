# python test-ckbd-sun.py --split depth --gpu_id 4 --q 1 &
# python test-ckbd-sun.py --split depth --gpu_id 5 --q 2 &
# python test-ckbd-sun.py --split depth --gpu_id 6 --q 3 &
# python test-ckbd-sun.py --split depth --gpu_id 7 --q 4 &


# python test-ckbd-sun.py --split rgb --gpu_id 4 --q 2 
# python test-ckbd-sun.py --split rgb --gpu_id 4 --q 3 
# python test-ckbd-sun.py --split rgb --gpu_id 4 --q 4 
# python test-ckbd-sun.py --split rgb --gpu_id 4 --q 5 


# python test-ckbd.py --split depth --gpu_id 5 --q 1 
# python test-ckbd.py --split depth --gpu_id 5 --q 2 
# python test-ckbd.py --split depth --gpu_id 5 --q 3 
# python test-ckbd.py --split depth --gpu_id 5 --q 4 


# python test-ckbd.py --split rgb --gpu_id 7 --q 2 
# python test-ckbd.py --split rgb --gpu_id 7 --q 3 
# python test-ckbd.py --split rgb --gpu_id 7 --q 4 
# python test-ckbd.py --split rgb --gpu_id 7 --q 5 



# python test-sun.py --split depth --gpu_id 0 --q 1 & 
# python test-sun.py --split depth --gpu_id 1 --q 2 &
# python test-sun.py --split depth --gpu_id 2 --q 3 &
# python test-sun.py --split depth --gpu_id 3 --q 4 &

# python test.py --split depth --gpu_id 0 --q 0 --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_dloss_0/checkpoints/checkpoint_latest.pth.tar & 
# python test.py --split depth --gpu_id 1 --q 1 --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_dloss_1/checkpoints/checkpoint_latest.pth.tar & 
# python test.py --split depth --gpu_id 2 --q 2 --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_dloss_2/checkpoints/checkpoint_latest.pth.tar & 
# python test.py --split depth --gpu_id 3 --q 3 --checkpoint  /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_dloss_3/checkpoints/checkpoint_latest.pth.tar & 


# python test.py --split rgb --gpu_id 6 --q 2 
# python test.py --split rgb --gpu_id 6 --q 3 
# python test.py --split rgb --gpu_id 6 --q 4 
# python test.py --split rgb --gpu_id 6 --q 5 

# python test-sun.py --split depth --gpu_id 6 --q 1 
# python test-sun.py --split depth --gpu_id 6 --q 2 
# python test-sun.py --split depth --gpu_id 6 --q 3 
# python test-sun.py --split depth --gpu_id 6 --q 4 


# python test-sun.py --split rgb --gpu_id 6 --q 2 
# python test-sun.py --split rgb --gpu_id 6 --q 3 
# python test-sun.py --split rgb --gpu_id 6 --q 4 
# python test-sun.py --split rgb --gpu_id 6 --q 5 




# python test.py --split rgb --gpu_id 3 --q 1 
# python test_united4.py --gpu_id 1 --q 5_5 -m ELIC_cpf_dec &
# python test_united4.py --gpu_id 2 --q 2_2 -m ELIC_cpf_dec &
# python test_united4.py --gpu_id 3 --q 3_3 -m ELIC_cpf_dec &
# python test_united4.py --gpu_id 4 --q 4_4 -m ELIC_cpf_dec &
# python test_united.py --gpu_id 6 --q 5_4 -m ELIC_cpf 
# python test_united.py --gpu_id 1 --q 2_2 -m ELIC_cpf
# python test_united.py --gpu_id 6 --q 5_5 -m ELIC_cpf
# python test_united.py --gpu_id 6 --q 4_6 -m ELIC_cpf
# python test_united.py --gpu_id 3 --q 2_1 -m ELIC_cpf
# python test_united.py --gpu_id 4 --q 2_3 -m ELIC_cpf
# python test_united.py --gpu_id 4 --q 4_3 -m ELIC_cpf
# python test_united.py --gpu_id 6 --q 5_5 -m ELIC_cpf&  
# python test_united.py --gpu_id 3 --q 5_5 -m ELIC_cpf1
# python test_united.py --gpu_id 6 --q 4.5_4.5 -m ELIC_cpf

# python test_united.py --gpu_id 6 --q 2_2 -m ELIC_EEM
# python test_united.py --gpu_id 1 --q 2_3 -m ELIC_EEM
# python test_united.py --gpu_id 6 --q 2_4 -m ELIC_EEM
# python test_united.py --gpu_id 6 --q 3_5 -m ELIC_EEM
# python test_united.py --gpu_id 6 --q 4_4 -m ELIC_EEM
# python test_united.py --gpu_id 6 --q 4_6 -m ELIC_EEM
# python test_united.py --gpu_id 3 --q 5_7 -m ELIC_EEM

# python test_united.py --gpu_id 6 --q 2_1 -m ELIC_united # 
#python test_united.py --gpu_id 3 --q 2_2 -m ELIC_united
#python test_united.py --gpu_id 4 --q 2_3 -m ELIC_united1
# python test_united.py --gpu_id 6 --q 3_3 -m ELIC_united
# python test_united.py --gpu_id 6 --q 4_3 -m ELIC_united
# python test_united.py --gpu_id 6 --q 4_4 -m ELIC_united
# python test_united.py --gpu_id 6 --q 4.5_4.5 -m ELIC_united
# python test_united.py --gpu_id 6 --q 5_4 -m ELIC_united # 
# python test_united.py --gpu_id 6 --q 2_2.5 -m ELIC_united1 
# python test_united.py --gpu_id 3 --q 3_4 -m ELIC_united
# python test_united.py --gpu_id 6 --q 5_5 -m ELIC_united
# python test_united.py --gpu_id 6 --q 5_5 -m ELIC_cpf

# python test_united.py --gpu_id 6 --q 2_2 -m ELIC_EEM

# python test_united.py --gpu_id 6 --q 4_6 -m ELIC_cpf

# python test_united.py --gpu_id 4 --q 5_5.5 -m ELIC_united 
# python test_united.py --gpu_id 5 --q 4_4.5 -m ELIC_united

# python test_united.py --gpu_id 3 --q 5_5 -m ELIC_cpf2 


# python test_united.py --gpu_id 5 --q 2_2 -m ELIC_cpf 
# python test_united.py --gpu_id 5 --q 3_3 -m ELIC_cpf 
# python test_united.py --gpu_id 5 --q 4_4 -m ELIC_cpf
# python test_united.py --gpu_id 5 --q 5_5 -m ELIC_cpf 

# python test_united4.py --gpu_id 6 --q 3_3.5 -m ELIC_united_lh
# python test_united4.py --gpu_id 6 --q 4_4.5 -m ELIC_united_lh
# python test_united4.py --gpu_id 6 --q 5_5.5 -m ELIC_united_lh
# python test_united4.py --gpu_id 6 --q 2.5_3 -m ELIC_united_lh

# python test_united4.py --gpu_id 6 --q 2_2.25 -m ELIC_united_LH 
# python test_united4.py --gpu_id 6 --q 3_3.25 -m ELIC_united_LH 
# python test_united4.py --gpu_id 6 --q 2_2 -m ELIC_united4_SE_res 
# python test_united4.py --gpu_id 6 --q 3_3 -m ELIC_united4_SE_res 
# python test_united4.py --gpu_id 6 --q 4_4 -m ELIC_united4_SE_res 
# python test_united4.py --gpu_id 6 --q 5_5 -m ELIC_united4_SE_Res 
# # python test_united.py --gpu_id 6 --q 5_5 -m ELIC_cpf 


# python test_united4.py --gpu_id 6 --q 2_2.25 -m ELIC_EEM_lh 
# python test_united4.py --gpu_id 6 --q 3_3.25 -m ELIC_EEM_lh 
# python test_united4.py --gpu_id 6 --q 2_2 -m ELIC_EEM_SE_res 
# python test_united4.py --gpu_id 6 --q 3_3 -m ELIC_EEM_SE_res 
# python test_united4.py --gpu_id 6 --q 4_4 -m ELIC_EEM_SE_res 
# python test_united4.py --gpu_id 6 --q 5_5 -m ELIC_EEM_SE_Res 

# python test_united4.py --gpu_id 6 --q 2_2 -m ELIC_cpf400 
# python test_united4.py --gpu_id 6 --q 3_3 -m ELIC_cpf400 
# python test_united4.py --gpu_id 6 --q 4_4 -m ELIC_cpf400 
# python test_united4.py --gpu_id 6 --q 5_5 -m ELIC_cpf400 

# python test_united4.py --gpu_id 0 --q 5_5 -m ELIC_cpf_dec &
# python test_united4.py --gpu_id 1 --q 4_4 -m ELIC_cpf_dec 
# python test_united4.py --gpu_id 0 --q 3_3 -m ELIC_cpf_dec &
# python test_united4.py --gpu_id 1 --q 2_2 -m ELIC_cpf_dec

# python test_united4.py --gpu_id 0 --q 5_4.75 -m ELIC_cpf_dec &
# python test_united4.py --gpu_id 1 --q 4_3.75 -m ELIC_cpf_dec 
# python test_united4.py --gpu_id 0 --q 3_2.75 -m ELIC_cpf_dec &
# python test_united4.py --gpu_id 1 --q 2_1.75 -m ELIC_cpf_dec

python test_united4.py --gpu_id 0 --q 5_5.5 -m ELIC_united-0.5 & 
python test_united4.py --gpu_id 1 --q 4_4.5 -m ELIC_united-0.5 
python test_united4.py --gpu_id 0 --q 3_3.5 -m ELIC_united-0.5 &
python test_united4.py --gpu_id 1 --q 2_2.5 -m ELIC_united-0.5 

python test_united4.py --gpu_id 0 --q 5_6 -m ELIC_united-1 & 
python test_united4.py --gpu_id 1 --q 4_5 -m ELIC_united-1 
python test_united4.py --gpu_id 0 --q 3_4 -m ELIC_united-1 &
python test_united4.py --gpu_id 1 --q 2_3 -m ELIC_united-1 

# python test_united4.py --gpu_id 5 --q 5_5.25 -m ELIC_cpf_enc_sp &
# python test_united4.py --gpu_id 6 --q 4_4.25 -m ELIC_cpf_enc_sp 
# python test_united4.py --gpu_id 3 --q 3_3.25 -m ELIC_cpf_enc_sp &
# python test_united4.py --gpu_id 4 --q 2_2.25 -m ELIC_cpf_enc_sp 

# python test_united4-sun.py --gpu_id 2 --q 5_5 -m ELIC_united_lh
# python test_united4-sun.py --gpu_id 2 --q 4_4 -m ELIC_united_lh
# python test_united4-sun.py --gpu_id 2 --q 3_3 -m ELIC_united_lh
# python test_united4-sun.py --gpu_id 2 --q 2_2 -m ELIC_united_lh


# python test_master.py --gpu_id 2 --q 4 -m ELIC_master --checkpoint /data/chenminghui/ELIC/experiments/nyuv2_depth_ELIC_master_4/checkpoints/checkpoint_latest.pth.tar  --checkpoint1 /data/chenminghui/ELIC/experiments/nyuv2_rgb_ELIC_5/checkpoints/checkpoint_epoch399.pth.tar


