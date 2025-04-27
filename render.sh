export CUDA_VISIBLE_DEVICES=1 \
subjects=(074 104 165 175 210 218 264 302 304) # 
for SUBJECT in "${subjects[@]}"; do
    python render.py \
	-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
	--skip_train #--skip_val
done