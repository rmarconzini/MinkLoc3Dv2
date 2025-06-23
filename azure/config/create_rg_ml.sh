az group create --name ml-rg-france --location francecentral

az ml workspace create --name ml-ws-france \
  --resource-group ml-rg-france \
  --location francecentral