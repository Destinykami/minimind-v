from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="AISHELL/AISHELL-1",
    repo_type="dataset",
    local_dir="dataset/AISHELL-1"
)
