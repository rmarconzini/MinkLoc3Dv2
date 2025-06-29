# az ml job create \
#   --file job03_v1.yml \
#   --set environment_variables.WANDB_API_KEY=8c8ceb8c56193f5176817c8fda2dbe6b4922ab02 \
#   --set environment_variables.WANDB_MODE=online \
#   -g rg-ml-france \
#   -w ml-ws-france

# az ml job create \
#   --file job07_v1.yml \
#   --set environment_variables.WANDB_API_KEY=8c8ceb8c56193f5176817c8fda2dbe6b4922ab02 \
#   --set environment_variables.WANDB_MODE=online \
#   -g rg-ml-france \
#   -w ml-ws-france

# az ml job create \
#   --file job09_v1.yml \
#   --set environment_variables.WANDB_API_KEY=8c8ceb8c56193f5176817c8fda2dbe6b4922ab02 \
#   --set environment_variables.WANDB_MODE=online \
#   -g rg-ml-france \
#   -w ml-ws-france

az ml job create \
  --file job03_v2.yml \
  --set environment_variables.WANDB_API_KEY=8c8ceb8c56193f5176817c8fda2dbe6b4922ab02 \
  --set environment_variables.WANDB_MODE=online \
  -g rg-ml-france \
  -w ml-ws-france

az ml job create \
  --file job07_v2.yml \
  --set environment_variables.WANDB_API_KEY=8c8ceb8c56193f5176817c8fda2dbe6b4922ab02 \
  --set environment_variables.WANDB_MODE=online \
  -g rg-ml-france \
  -w ml-ws-france

az ml job create \
  --file job09_v2.yml \
  --set environment_variables.WANDB_API_KEY=8c8ceb8c56193f5176817c8fda2dbe6b4922ab02 \
  --set environment_variables.WANDB_MODE=online \
  -g rg-ml-france \
  -w ml-ws-france