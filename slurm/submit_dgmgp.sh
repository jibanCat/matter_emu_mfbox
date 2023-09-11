#!/bin/bash

# $1: data/folder to L1
# $2: data/folder to L2
# $3: data/folder to H
# $4: data/folder to test
# $5, 6: zmin, zmax
# $7, 8: num L1, num L2
# $9: hf selected ind

sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --job-name=GMGP_$7_L1-$1_$8_L2-$2_H-$3_zmin-$5_zmax-$6_hf-$9
#SBATCH -p batch
#SBATCH --output="GMGP_$7_L1-$1_$8_L2-$2_H-$3_zmin-$5_zmax-$6_hf-$9.out"

date

echo "----"
# run python script
python -u -c "from examples.optimize_mfemulator import *;\
optimize_mfemulator(\
lf_filename='data/$1/cc_emulator_powerspecs.hdf5',\
hf_filename='data/$3/cc_emulator_powerspecs.hdf5',\
test_filename='data/$4/cc_emulator_powerspecs.hdf5',\
lf_json='data/$1/emulator_params.json',\
hf_json='data/$3/emulator_params.json',\
test_json='data/$4/emulator_params.json',\
max_z=$6, min_z=$5,\
n_optimization_restarts=20,\
parallel=False,\
dGMGP=True,\
lf_filename_2='data/$2/cc_emulator_powerspecs.hdf5',\
lf_json_2='data/$2/emulator_params.json',\
num_lf=$7,\
num_lf_2=$8,\
hf_selected_ind=$9, )"


hostname

exit

EOT
