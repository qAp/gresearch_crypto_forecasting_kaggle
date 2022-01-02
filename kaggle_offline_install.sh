pip install /kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/axial_positional_embedding-0.2.1.tar.gz
pip install /kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/local_attention-1.4.3-py3-none-any.whl
pip install /kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/einops-0.3.2-py3-none-any.whl
pip install /kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/antlr4-python3-runtime-4.8.tar.gz

pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ Cython>=0.22
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ cmdstanpy
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ pystan~=2.19.1.1
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ numpy>=1.15.4
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ convertdate>=2.1.2
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ python-dateutil>=2.8.0
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ performer-pytorch
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/  nystrom-attention
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ omegaconf
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ tqdm>=4.36.1
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ pytorch-lightning==1.3.8
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ scikit-learn==0.23.2

cd /kaggle
cp -r /kaggle/input/stfdata/gresearch_crypto_forecasting_kaggle . 
cd /kaggle/gresearch_crypto_forecasting_kaggle/
pip install --no-index --no-deps --find-links=/kaggle/input/stf00-offline-requirements/spacetimeformer_requirements/ -e .