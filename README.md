# Background 
Explore using GAN to design a room layout that's energy efficient

# Set up 
```
# Setup conda environment
conda create -n pix2pix
conda activate pix2pix
# pip list --format=freeze > requirements.txt

# Setup dependencies
pip install -r requirements.txt
```

This is built on https://github.com/nate-peters/pix2pix-floorplans-dataset
# pix2pix-floorplans-dataset


### About the data:
The dataset is derived from images of floor plans for small, single-story houses sourced from HousePlans.com. Each image was manually tagged using Rhino and Grasshopper, and exported as an A/B image pair for use with pix2pix. The A image contains the boundary shape of each floor plan represented as a solid black region. The B image contains the same boundary, but with color coded regions that correspond to different room types in the plan.

![](extra/example-AB-pair.png)

### Organization of the `/dataset` directory:

This directory contains the images that we'll use to train a pix2pix model.
`Source Images`: These are the source images that informed the room tags.  
`Original`: 
- `/A`: The boundary shape of each floor plan represented as a solid black region
- `/B`: The color tagged plans
