echo "Loading Python module..."
module load python/3.3

echo "Creating and/or setting Python path..."
mkdir -p $HOME/python/lib64/python3.3/site-packages
export PYTHONPATH=$HOME/python/lib64/python3.3/site-packages:$PYTHONPATH

echo "Installing necessary packages..."
pip3 install --install-option="--prefix=$HOME/python" networkx joblib

echo "Done!"