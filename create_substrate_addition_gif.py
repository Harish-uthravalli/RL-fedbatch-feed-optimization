import imageio
import glob

def create_gif(image_folder, output_gif):
    images = []
    files = sorted(glob.glob(f"{image_folder}/*.png"))  # Sort frames by order
    for filename in files:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_gif, images, fps=30, loop=0)  # Adjust FPS if needed

# Generate GIFs
create_gif("substrate_addition", "gifs/s_add.gif")