algo='segnext-B'
meta_dir='segnext-B'

config="configs/main/${algo}.py"
checkpoint="work_dirs/${meta_dir}/best_mIoU_iter_13500.pth"
out="work_dirs/${meta_dir}/pred/"
work_dir="work_dirs/${meta_dir}/"
show_dir="work_dirs/${meta_dir}/"


python3 tools/test.py --config $config --work-dir $work_dir  --checkpoint $checkpoint --out $out --show-dir $show_dir