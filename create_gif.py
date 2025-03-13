import imageio
import glob
import os

def create_gif(image_folder, output_gif):
    images = []
    files = sorted(glob.glob(f"{image_folder}/*.png"))  # Sort frames by order
    for filename in files:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_gif, images, fps=30)  # Adjust FPS if needed

# Generate GIFs
#create_gif("xse_plots", "gifs/xse_animation.gif")
image_folder = os.path.join('perturbation_images','pump','xse')
create_gif(image_folder, "gifs/pump_pert_xse.gif")
