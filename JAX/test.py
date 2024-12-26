import pickle

with open('/home/bytetriper/model_zoo/mae_jax.pkl', 'rb') as f:
    params = pickle.load(f)['params']

    print(params['default_id_restore'].dtype)
