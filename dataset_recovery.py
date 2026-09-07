import os
import shutil

def move_files(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_file = os.path.join(root, file)
            relative_path = os.path.relpath(source_file, source_dir)
            target_file = os.path.join(target_dir, relative_path)

            os.makedirs(os.path.dirname(target_file), exist_ok=True)

            shutil.move(source_file, target_file)
            print(f"Moved: {source_file} -> {target_file}")

if __name__ == "__main__":
    source_dir = "/data/StereoDatasets/sceneflow_rltrain/"
    target_dir = "/data/StereoDatasets/sceneflow/"

    move_files(source_dir, target_dir)

    source_dir = "/data/StereoDatasets/sceneflow_rltest/"
    target_dir = "/data/StereoDatasets/sceneflow/"

    move_files(source_dir, target_dir)