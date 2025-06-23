az ml job create \
  --file job.yml \
  --set environment_variables.WANDB_API_KEY= \
  --set environment_variables.WANDB_MODE=online \
  -g rg-ml-france \
  -w ml-ws-france