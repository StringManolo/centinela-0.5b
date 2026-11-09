gh workflow run build-r2vul-gguf.yml \
  --ref r2vul-1.5b \
  -f zenodo_orpo_url="https://zenodo.org/records/16741648/files/orpo.zip?download=1" \
  -f base_model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  -f release_tag="r2vul-1.5b-gguf-v1"
