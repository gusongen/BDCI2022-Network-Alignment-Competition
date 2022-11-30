# python train.py --graph_s data_G1 --graph_d data_G2 --anoise 0.2
# python train.py --graph_s data_G1 --graph_d data_G2 --anoise 0.4
python train.py --graph_s data_G1 --graph_d data_G3 --anoise 0.2
# python train.py --graph_s data_G1 --graph_d data_G3 --anoise 0.4

# 将四个文件写入一个文件
# cat submit_tmp_data_G1_data_G2_0.2.txt \
#     submit_tmp_data_G1_data_G2_0.4.txt \
#     submit_tmp_data_G1_data_G3_0.2.txt \
#     submit_tmp_data_G1_data_G3_0.4.txt > submit.txt