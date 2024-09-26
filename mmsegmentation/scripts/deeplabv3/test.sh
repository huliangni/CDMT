algo='deeplabv3'
meta_dir='deeplabv3'

config="configs/main/${algo}.py"
checkpoint="work_dirs/${meta_dir}/iter_20000.pth"
out="work_dirs/${meta_dir}/pred/"
work_dir="work_dirs/${meta_dir}/"
show_dir="work_dirs/${meta_dir}/"


python3 tools/test.py --config $config --work-dir $work_dir  --checkpoint $checkpoint --out $out --show-dir $show_dir