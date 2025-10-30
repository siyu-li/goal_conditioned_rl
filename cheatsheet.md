### uninstall cuda 13 and install cuda 12
# remove cuda meta-packages and related tools (be careful)
sudo apt-get remove --purge '^cuda.*' 'nvidia-cuda-toolkit' 'libcudnn*' 'nsight*' -y
sudo apt-get autoremove -y
# remove remaining files (optional)
sudo rm -rf /usr/local/cuda* /usr/local/cuda-13*
# Example for Ubuntu (replace with NVIDIA-provided repo and specific cuda-12.X package per their guide)
# 1) Add NVIDIA package repository (commands vary by Ubuntu version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

# 2) Install CUDA 12 toolkit (meta-package)
sudo apt-get -y install cuda-toolkit-12-1   # replace 12-1 with the exact package you want (12.0/12.1/12.2)


# 1) Create or activate conda env
conda activate <your-env>

# 3) Remove CPU jaxlib and reinstall jax + cuda12 jaxlib (pinned to 0.4.38)
pip uninstall -y jax jaxlib
pip install jax==0.4.33
pip install --upgrade "jax[cuda12_pip]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4) verify
python -c "import jax, jaxlib; print('jax', jax.__version__); print('jaxlib', jaxlib.__version__); print('backend', jax.default_backend()); print('devices', jax.devices())"

python train.py n_epochs=10 agent=sac env_name=CentralizedTwoRobot-v1 hindsight=her agent.critic.dropout=0.0

python train.py n_epochs=10 agent=sac env_name=CentralizedFourRobot-v2 hindsight=her agent.critic.dropout=0.0

python train.py n_epochs=10 agent=sac env_name=CentralizedTwoRobot-v1_5 hindsight=her agent.critic.dropout=0.0

python train.py n_epochs=1000 agent=sac env_name=CentralizedThreeRobot-v2_3r hindsight=her agent.critic.dropout=0.0

python demo.py --demo_path parker/CentralizedFourRobot-v2/sac_16-36-27
python demo.py --demo_path parker/CentralizedTwoRobot-v1_5/sac_13-53-33
python demo.py --demo_path parker/CentralizedThreeRobot-v2_3r/sac_15-04-39


pip install numpy 1.26.3
pip install jax==0.4.33
pip install --upgrade "jax[cuda12_pip]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip cache purge