#!/bin/bash

# Check if the run name argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <run-name>"
    exit 1
fi

# Variables
RUN_NAME="$1"
S3_URL="s3://jordanai-checkpoints/$RUN_NAME/output/model.tar.gz"
REMOTE_USER="lancelot"
REMOTE_HOST="matlaber8"
LEVANTER_DIR="/u/$REMOTE_USER/levanter"
REMOTE_DIR="$LEVANTER_DIR/checkpoints/$RUN_NAME"
REMOTE_STEP_FILE="$REMOTE_DIR/step_number.txt"  # Temporary file to store the step number
LOCAL_ANTICIPATION_DIR="$HOME/Desktop/MediaLabMAS/AI Music Repos/anticipation"
LOCAL_CHECKPOINT_DIR="$LOCAL_ANTICIPATION_DIR/checkpoints"
ONNX_OUTPUT_DIR="$HOME/Downloads"

# 1. Connect via SSH and download the checkpoint from S3
ssh $REMOTE_USER@$REMOTE_HOST << EOF
    # Ensure remote directory exists
    mkdir -p $REMOTE_DIR
    cd $REMOTE_DIR

    # Download checkpoint from S3
    aws s3 cp $S3_URL .

    # 2. Untar the checkpoint
    tar -xzf $(basename $S3_URL)

    # 3. Find the extracted directory (ignoring the .tar.gz file)
    EXTRACTED_FOLDER_NAME=\$(ls -d */ | grep -v "$(basename $S3_URL)")

    # Enter the extracted directory
    cd "\$EXTRACTED_FOLDER_NAME"

    # 4. Find the latest checkpoint
    LATEST_CHECKPOINT=\$(ls -d step-* | sort -V | tail -n 1)

    # Extract the step number from the latest checkpoint's directory name
    STEP_NUMBER=\$(echo "\$LATEST_CHECKPOINT" | grep -o '[0-9]\+')

    # Save the step number to a temporary file for later retrieval
    echo \$STEP_NUMBER > $REMOTE_STEP_FILE

    # Move the latest checkpoint to the parent directory
    mv \$LATEST_CHECKPOINT ..
    cd ..
    rm -rf "\$EXTRACTED_FOLDER_NAME" $(basename $S3_URL)

    cd $LEVANTER_DIR

    # 5. Activate the virtual environment and run the Python script
    source .venv/bin/activate
    python -m levanter.main.export_lm_to_hf --model.type gpt2 --model.hidden_dim 1024 --model.num_heads 16 --model.num_layers 24 --model.seq_len 1024 --model.scale_attn_by_inverse_layer_idx true --model.gradient_checkpointing true --model.use_flash_attention false --override_vocab_size 55028 --output_dir $REMOTE_DIR/\$LATEST_CHECKPOINT/hf --checkpoint_path $REMOTE_DIR/\$LATEST_CHECKPOINT/ --save_tokenizer false

    # 6. Move to the output directory and zip it
    cd $REMOTE_DIR/\$LATEST_CHECKPOINT
    zip -r ~/${RUN_NAME}_\${STEP_NUMBER}.zip hf
EOF

# 7. Retrieve the step number from the remote server and store it locally
STEP_NUMBER=$(ssh $REMOTE_USER@$REMOTE_HOST "cat $REMOTE_STEP_FILE")

# 8. Download the zip file to the local machine
scp $REMOTE_USER@$REMOTE_HOST:~/${RUN_NAME}_${STEP_NUMBER}.zip "$LOCAL_CHECKPOINT_DIR/${RUN_NAME}_${STEP_NUMBER}.zip"

# 9. Unzip the checkpoint on the local machine
unzip "$LOCAL_CHECKPOINT_DIR/${RUN_NAME}_${STEP_NUMBER}.zip" -d "$LOCAL_CHECKPOINT_DIR"

# Move the checkpoint to the correct directory
mv "$LOCAL_CHECKPOINT_DIR/hf" "$LOCAL_CHECKPOINT_DIR/${RUN_NAME}_${STEP_NUMBER}"

# Delete the zip file
rm "$LOCAL_CHECKPOINT_DIR/${RUN_NAME}_${STEP_NUMBER}.zip"

# 10. Run the ONNX conversion using the local Python virtual environment
source "$LOCAL_ANTICIPATION_DIR/experiments/.venv/bin/activate"
python3 -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m "$LOCAL_CHECKPOINT_DIR/${RUN_NAME}_${STEP_NUMBER}" --output "$ONNX_OUTPUT_DIR/${RUN_NAME}_${STEP_NUMBER}.onnx" -o -p fp32 -t 10 >"$ONNX_OUTPUT_DIR/export_output.txt" 2>&1

# Delete the past ONNX file
rm "$ONNX_OUTPUT_DIR/${RUN_NAME}_${STEP_NUMBER}_past.onnx"

# Deactivate the virtual environment
deactivate

# Delete the step number file on the remote server
ssh $REMOTE_USER@$REMOTE_HOST "rm $REMOTE_STEP_FILE"

echo "ONNX conversion complete: $ONNX_OUTPUT_DIR/${RUN_NAME}_${STEP_NUMBER}.onnx. Check $ONNX_OUTPUT_DIR/export_output.txt for any errors."