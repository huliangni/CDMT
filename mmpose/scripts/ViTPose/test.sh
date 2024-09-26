algo='simple'
meta_dir='simple'

config="configs/main/${algo}.py"
checkpoint="work_dirs/${meta_dir}/best_NME_epoch_209.pth"
dump="work_dirs/${meta_dir}/pred.pkl"
work_dir="work_dirs/${meta_dir}/"
show_dir="work_dirs/${meta_dir}/"

cd ..
cd ..
python3 tools/test.py --config $config --work-dir $work_dir  --checkpoint $checkpoint --dump $dump --show-dir $show_dir