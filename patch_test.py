from huggingface_hub import list_repo_files
print("Files in h94/IP-Adapter-FaceID:")
for f in list_repo_files("h94/IP-Adapter-FaceID"):
    print(f)
