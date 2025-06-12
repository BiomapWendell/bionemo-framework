# Enable fused attention in transformer engine for speed-up
# This is a demo run
DATA_DIR=$(download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source ngc)

train_esm2 \
    --train-cluster-path ${DATA_DIR}/2024_03/train_clusters.parquet \
    --train-database-path ${DATA_DIR}/2024_03/train.db \
    --valid-cluster-path ${DATA_DIR}/2024_03/valid_clusters.parquet \
    --valid-database-path ${DATA_DIR}/2024_03/validation.db \
    --precision="bf16-mixed" \
    --num-gpus 1 \
    --num-nodes 1 \
    --num-steps 50000 \
    --val-check-interval 500 \
    --result-dir ./results/esm2-demo/ \
    --max-seq-length 1024 \
    --resume-if-exists \
    --limit-val-batches 4 \
    --micro-batch-size 8 \
    --num-layers 12 \
    --hidden-size 480 \
    --num-attention-head 20 \
    --ffn-hidden-size 1920 \
    --tensor-model-parallel-size 1 \
    --create-tensorboard-logger \
    # --wandb-project "test" \
    # --wandb-entity "test" \
    # --experiment-name "test" \
    # --num-dataset-workers 8
