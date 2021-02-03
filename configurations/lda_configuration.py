"""See https://radimrehurek.com/gensim/models/ldamulticore.html for more info on parameters"""
lda_model = {
    'model_parameters': {
        'workers': 4,
        'chunksize': 20,
        'passes': 10,
        'eval_every': 10,
        'random_state': 100,
        'minimum_probability': 0.01,
        'minimum_phi_value': 0.01,
        'per_word_topics': True
    }
}
