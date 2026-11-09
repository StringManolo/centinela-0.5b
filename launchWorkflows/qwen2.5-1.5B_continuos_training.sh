gh workflow run centinela-optimized-training.yml \
  --ref qwen25-1.5b \
  -f base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  -f model_size="1.5B" \
  -f techniques="rslora+rank32-qwen25" \
  -f num_epochs="3" \
  -f time_budget_min="280" \
  -f sec_per_step="45" \
  -f resume_from="latest"
