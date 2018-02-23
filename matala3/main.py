from models.super_resolution import create_model

sr_model = create_model()
print(sr_model.summary())