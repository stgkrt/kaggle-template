# Description: Run the experiment
exp_name="default"
# /kaggle/working/にexp_nameのディレクトリがあるときsuffixをつける
if [ -d /kaggle/working/$exp_name ]; then
  suffix=$(date "+_%Y%m%d_%H%M%S")
  exp_name=$exp_name$suffix
fi
echo "exp_name: $exp_name"
# train
python src/train.py exp_name="$exp_name" splits=fold0
python src/train.py exp_name="$exp_name" splits=fold1
