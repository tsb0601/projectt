#define a batch of experiments to run
experiments=('./ckpt_gcs/DiT_B_1_mae/XL2_b2048_ema/SIDXL_cosine_h1024_d24_ep80_lr8e-4_b2048_ema99_sigmoid_07112024_231336') # './ckpt_gcs/DiT_B_1_mae/XL2_b2048_ema/SIDXL_cosine_h1024_d24_ep80_lr8e-4_b2048_ema99_sigmoid_gembed_08112024_232503')
gen_names=('sigmoid' 'sigmoid_gembed')
# for i in range(0, len(experiments)), run the following command
# ./infer_stage2.sh ./ckpt/DiT_XL_1_mae_gen/$gen_name[i] $experiments[i]/ep_last-checkpoint $experiments[i]/config.yaml 4

for i in {0..1}
do
    echo "Running inference for ${experiments[i]}"
    ./infer_stage2.sh ./ckpt/DiT_XL_1_mae_gen/${gen_names[i]} ${experiments[i]}/ep_last-checkpoint ${experiments[i]}/config.yaml 4
done