import os
import urllib.request
import gzip
import shutil

# ✅ 使用 Google 提供的镜像（Yann LeCun 链接已失效）
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

os.makedirs("data", exist_ok=True)

for filename in files:
    url = base_url + filename
    out_path = os.path.join("data", filename)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, out_path)

    # 解压
    with gzip.open(out_path, 'rb') as f_in:
        with open(out_path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"✅ Saved and extracted: {out_path[:-3]}")

print("✅ MNIST dataset downloaded and extracted successfully.")
