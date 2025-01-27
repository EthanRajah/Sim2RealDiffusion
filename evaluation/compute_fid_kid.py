import os
from torch_fidelity import calculate_metrics
from PIL import Image

def resize_images(directory):
    """
    Resize images in the given directory to desired resolution if they are not already that size, prior to computing KID and FID scores.

    Args:
        directory (str): Path to the directory containing images.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    if img.size != (768, 512):
                        img_resized = img.resize((768, 512), Image.Resampling.LANCZOS)
                        img_resized.save(file_path)
            except Exception as e:
                print(f"Error processing '{filename}': {e}")

def compute_kid_fid(sim_dir, real_dir):
    """
    Compute KID and FID scores between two image directories.

    Args:
        sim_dir (str): Path to the directory containing realistic simulation images.
        real_dir (str): Path to the directory containing real images.
    """
    if not os.path.exists(sim_dir):
        raise FileNotFoundError(f"Simulation directory '{sim_dir}' does not exist.")
    if not os.path.exists(real_dir):
        raise FileNotFoundError(f"Real directory '{real_dir}' does not exist.")

    metrics = calculate_metrics(
        input1=sim_dir,
        input2=real_dir,
        cuda=True, 
        isc=False, 
        fid=True, 
        kid=True, 
        prc=False, 
        verbose=True,
        kid_subset_size=499  # Adjust subset size as needed
    )

if __name__ == "__main__":
    # Specify the paths to your directories
    #simulation_images_dir = "/home/ethan/DiffusionResearch/Sim2RealDiffusion/evaluation/sim_solid_pushblock_gen"
    #simulation_images_dir = "/home/ethan/DiffusionResearch/Sim2RealDiffusion/evaluation/sim_solid_pushblock_e16/solid_pushblock_e16"
    #real_images_dir = "/home/ethan/DiffusionResearch/experiments/solid_pushblock/preprocessed_data/e16/"
    #simulation_images_dir = "/home/ethan/DiffusionResearch/Sim2RealDiffusion/evaluation/sim_tissue_pushblock_e12/tissue_pushblock_e12"
    #real_images_dir = "/home/ethan/DiffusionResearch/experiments/tissue_pushblock/e12"
    #simulation_images_dir = "/home/ethan/DiffusionResearch/Sim2RealDiffusion/evaluation/sim_tissue_pushblock_gen"
    simulation_images_dir = "/home/ethan/DiffusionResearch/Sim2RealDiffusion/evaluation/sim_rope_gen"
    real_images_dir = "/home/ethan/DiffusionResearch/experiments/rope/500"

    # Resize real images if needed
    resize_images(real_images_dir)
    resize_images(simulation_images_dir)
    
    compute_kid_fid(simulation_images_dir, real_images_dir)

