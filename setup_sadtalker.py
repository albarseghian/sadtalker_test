import os
import sys
import subprocess
import requests
from tqdm import tqdm

def run_command(command, cwd=None):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    return True

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sadtalker_dir = os.path.join(project_root, "SadTalker")

    # 1. Clone SadTalker repository
    if not os.path.exists(sadtalker_dir):
        print("Cloning SadTalker repository...")
        if not run_command(["git", "clone", "https://github.com/OpenTalker/SadTalker.git"], cwd=project_root):
            return
    else:
        print("SadTalker repository already exists.")

    # 2. Install dependencies
    print("Installing SadTalker dependencies...")
    pip_path = os.path.join(os.path.dirname(sys.executable), "pip.exe")
    if not os.path.exists(pip_path):
        pip_path = "pip"
    
    run_command([pip_path, "install", "torchaudio", "face-alignment", "imageio", "scikit-image", "resampy", "kornia", "facexlib", "yacs", "gfpgan", "safetensors"], cwd=project_root)

    # 3. Download weights
    checkpoint_dir = os.path.join(sadtalker_dir, "checkpoints")
    gfpgan_dir = os.path.join(sadtalker_dir, "gfpgan", "weights")
    
    models = {
        "SadTalker_V0.0.2_256.safetensors": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
        "mapping_00109-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
        "auido2exp_00062-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00062-model.pth.tar",
        "auido2pose_00140-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2pose_00140-model.pth.tar",
        "facevid2vid_00189-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar",
        "epoch_20.pth": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/epoch_20.pth",
        "shape_predictor_68_face_landmarks.dat": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/shape_predictor_68_face_landmarks.dat",
    }

    for name, url in models.items():
        download_file(url, os.path.join(checkpoint_dir, name))

    # GFPGAN weights
    download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        os.path.join(gfpgan_dir, "GFPGANv1.4.pth")
    )

    # 4. Download facexlib models
    facex_dir = os.path.join(project_root, ".venv311", "Lib", "site-packages", "facexlib", "weights")
    os.makedirs(facex_dir, exist_ok=True)
    
    facex_models = {
        "detection_Resnet50_Final.pth": "https://huggingface.co/sczhou/CodeFormer/resolve/main/weights/facexlib/detection_Resnet50_Final.pth",
        "parsing_parsenet.pth": "https://huggingface.co/sczhou/CodeFormer/resolve/main/weights/facexlib/parsing_parsenet.pth",
    }
    
    for name, url in facex_models.items():
        download_file(url, os.path.join(facex_dir, name))

    print("\nSetup complete! SadTalker is ready for testing.")

if __name__ == "__main__":
    main()
