# set KAGGLE_KEY as environment variable before running training job
gcloud ai-platform local train \
  --package-path trainer \
  --module-name trainer.task \
  --job-dir results \
  -- \
  --kaggle-key $KAGGLE_KEY \
  --num-epochs 15 \
  --batch-size 128 \
  --learning-rate .01