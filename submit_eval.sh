az ml job create \
  --file job_evaluate_der.yml \
  --set environment_variables.WANDB_API_KEY=8c8ceb8c56193f5176817c8fda2dbe6b4922ab02 \
  --set environment_variables.WANDB_MODE=online \
  -g rg-ml-france \
  -w ml-ws-france