#!/usr/bin/env bash
docker run \
    -v $PWD/src/scripts/conversion:/home/icecube/conversion \
    -v $PWD/data:/home/icecube/data \
    -u $(id -u):$(id -g) \
    --env-file /home/mads/repos/CubeML/.env \
    -it ehrhorn/i3-convert:latest \
    python /home/icecube/conversion/i3_to_hdf5_calc.py \
    -i 'MuonGun_Level2_139008' -n 0