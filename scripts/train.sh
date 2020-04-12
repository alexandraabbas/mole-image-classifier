timestamp=$(date +%d%m%Y_%H%M%S)

JOB_NAME="tf_mole_classifier_$timestamp"
# set BUCKET_NAME as environment variable before running training job
JOB_DIR="gs://$BUCKET_NAME/$JOB_NAME"
REGION="europe-west1"

gcloud ai-platform jobs submit training $JOB_NAME \
  --package-path trainer/ \
  --module-name trainer.task \
  --region $REGION \
  --python-version 3.7 \
  --runtime-version 2.1 \
  --job-dir $JOB_DIR \
  --stream-logs \
  -- \
  --kaggle-key $KAGGLE_KEY \
  --num-epochs 15 \
  --batch-size 128 \
  --learning-rate .01