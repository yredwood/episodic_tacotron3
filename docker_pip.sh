pip install numpy scipy matplotlib ipython jupyter pandas sympy nose scikit-learn torch torchvision tensorflow-gpu
grep -v '^#' requirements.txt | xargs -L 1 pip install --no-cache-dir
