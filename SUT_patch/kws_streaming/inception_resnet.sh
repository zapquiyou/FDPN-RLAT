CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
DATA_PATH=/data/gsc_splited

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

$CMD_TRAIN \
--batch_size 128 \
--split_data 0 \
--wanted_words 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes' \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir trained_inception_resnet_40000/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 40000,40000,40000,40000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
--save_step_interval 10000 \
--eval_step_interval 2000 \
--result_file_path '/data/gsc_splited/result/' \
inception_resnet \
--cnn1_filters '32' \
--cnn1_strides '1' \
--cnn1_kernel_sizes '5' \
--cnn2_scales '0.2,0.5,1.0' \
--cnn2_filters_branch0 '16,16,32' \
--cnn2_filters_branch1 '16,16,32' \
--cnn2_filters_branch2 '32,32,64' \
--cnn2_strides '2,2,1' \
--cnn2_kernel_sizes '3,5,5' \
--bn_scale 1 \
--dropout 0.0

$CMD_TRAIN \
--batch_size 128 \
--split_data 0 \
--wanted_words 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes' \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir trained_inception_resnet_38000/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 38000,38000,38000,38000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
--save_step_interval 10000 \
--eval_step_interval 2000 \
--result_file_path '/data/gsc_splited/result/' \
inception_resnet \
--cnn1_filters '32' \
--cnn1_strides '1' \
--cnn1_kernel_sizes '5' \
--cnn2_scales '0.2,0.5,1.0' \
--cnn2_filters_branch0 '16,16,32' \
--cnn2_filters_branch1 '16,16,32' \
--cnn2_filters_branch2 '32,32,64' \
--cnn2_strides '2,2,1' \
--cnn2_kernel_sizes '3,5,5' \
--bn_scale 1 \
--dropout 0.0

$CMD_TRAIN \
--batch_size 128 \
--split_data 0 \
--wanted_words 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes' \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir trained_inception_resnet_36000/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 36000,36000,36000,36000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
--save_step_interval 10000 \
--eval_step_interval 2000 \
--result_file_path '/data/gsc_splited/result/' \
inception_resnet \
--cnn1_filters '32' \
--cnn1_strides '1' \
--cnn1_kernel_sizes '5' \
--cnn2_scales '0.2,0.5,1.0' \
--cnn2_filters_branch0 '16,16,32' \
--cnn2_filters_branch1 '16,16,32' \
--cnn2_filters_branch2 '32,32,64' \
--cnn2_strides '2,2,1' \
--cnn2_kernel_sizes '3,5,5' \
--bn_scale 1 \
--dropout 0.0

$CMD_TRAIN \
--batch_size 128 \
--split_data 0 \
--wanted_words 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes' \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir trained_inception_resnet_34000/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 34000,34000,34000,34000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 80 \
--dct_num_features 30 \
--resample 0.15 \
--time_shift_ms 100 \
--train 1 \
--feature_type 'mfcc_op' \
--fft_magnitude_squared 1 \
--preprocess 'raw' \
--save_step_interval 10000 \
--eval_step_interval 2000 \
--result_file_path '/data/gsc_splited/result/' \
inception_resnet \
--cnn1_filters '32' \
--cnn1_strides '1' \
--cnn1_kernel_sizes '5' \
--cnn2_scales '0.2,0.5,1.0' \
--cnn2_filters_branch0 '16,16,32' \
--cnn2_filters_branch1 '16,16,32' \
--cnn2_filters_branch2 '32,32,64' \
--cnn2_strides '2,2,1' \
--cnn2_kernel_sizes '3,5,5' \
--bn_scale 1 \
--dropout 0.0