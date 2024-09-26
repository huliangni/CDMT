algo='darkpose'
meta_dir='darkpose'

config="configs/main/${algo}.py"
checkpoint="work_dirs/${meta_dir}/best_NME_epoch_20.pth"
dump="work_dirs/${meta_dir}/pred.pkl"
work_dir="work_dirs/${meta_dir}/"
show_dir="work_dirs/${meta_dir}/"


python3 tools/test.py --config $config --work-dir $work_dir  --checkpoint $checkpoint --dump $dump --show-dir $show_dir