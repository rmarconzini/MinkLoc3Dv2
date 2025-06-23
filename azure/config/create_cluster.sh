az ml compute create \
  --name T4x1-28G \
  --type AmlCompute \
  --size Standard_NC16as_T4_v3 \
  --min-instances 0 \
  --max-instances 1 \
  --resource-group rg-ml-france \
  --workspace-name ml-ws-france \
  --location francecentral