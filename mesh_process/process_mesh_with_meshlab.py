import os

script_path = './meshlab_script.mlx'
input_dir = '/media/administrator/Code/don/4DReconstruction/code/toy_experiment/data/doozy/mesh'
output_dir = '/media/administrator/Code/don/4DReconstruction/code/toy_experiment/data/doozy/mesh'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for name in os.listdir(input_dir):
    os.system('meshlabserver -i %s -o %s -s %s' % (
        os.path.join(input_dir, name), 
        os.path.join(output_dir, name), 
        script_path))

print('Done!')
