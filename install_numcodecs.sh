module load gcc

NUMCODECS_HOME=$PWD
git clone git@github.com:NCAR/numcodecs_sperr.git
cd numcodecs_sperr
git submodule update --init
cd SPERR
git checkout 495e31d68783f6cf55b4c7e33a9303774ceb2c53
module load cmake
cmake $PWD
make -j 4
cd $NUMCODECS_HOME
git clone https://github.com/shaomeng/MURaMKit.git
cd MURaMKit/
git checkout 7341a632e56170eae3f58e2c6e47072981c40887
cmake $PWD
make -j 4

cd $NUMCODECS_HOME
mkdir lib64
cp -P SPERR/src/lib*so* lib64/
cp -P MURaMKit/src/lib*so* lib64/
cp -P SPERR/zstd/install/lib/lib*so* lib64/

module load conda
conda activate mpich_sperr_de
mkdir -p ../pyenv/numcodecs-dev
PYTHON_EXEC=`which python`
virtualenv --no-site-packages --python=$PYTHON_EXEC ../pyenv/numcodecs-dev
source ../pyenv/numcodecs-dev/bin/activate
pip install setuptools_scm
pip install py-cpuinfo
pip install cython

source repair_wheel.sh

module load conda
conda activate derecho_sperr103
pip install dist/*whl --force-reinstall --no-deps


module load conda
conda env create -f enviroment.yml
conda activate mpich_sperr_de
env mpicc=`which mpicc` pip3 install mpi4py
pip install dist/*whl --force-reinstall --no-deps

