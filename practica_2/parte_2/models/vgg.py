import torch.nn as nn
from torchvision import models

def build_vgg(model_name: str, num_classes: int, freeze_until: int = 0):
    model_name = model_name.lower()
    vgg_models = {
        "vgg11": models.vgg11, #10
        "vgg13": models.vgg13, # 16
        "vgg16": models.vgg16, # 24
        "vgg19": models.vgg19# 28 
    }

    if model_name not in vgg_models:
        raise ValueError(f"Modelo {model_name} no soportado. Usa vgg11, vgg13, vgg16 o vgg19.")
    weights_attr = getattr(models, model_name.upper() + "_Weights")
    model = vgg_models[model_name](weights=weights_attr.IMAGENET1K_V1)
    if freeze_until is not None:
        for name, param in model.features.named_parameters():
            idx = int(name.split(".")[0])
            param.requires_grad = idx >= freeze_until

    # 3️⃣ Reemplazar la última capa del clasificador
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model

def build_vgg_extractor(model_name="vgg16", freeze_until=28, flatten=True):
    vgg_models = {
        "vgg11": models.vgg11,
        "vgg13": models.vgg13,
        "vgg16": models.vgg16,
        "vgg19": models.vgg19
    }
    weights_attr = getattr(models, model_name.upper() + "_Weights")
    model = vgg_models[model_name](weights=weights_attr.IMAGENET1K_V1)
    for i, (name, param) in enumerate(model.features.named_parameters()):
        param.requires_grad = i >= freeze_until
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool
    )

    if flatten:
        feature_extractor = nn.Sequential(
            feature_extractor,
            nn.Flatten()  
        )
    return feature_extractor
def build_vgg_extractor_fc(model_name="vgg16",  freeze_until=28,fc_layer_index=5, flatten=True):
    vgg_models = {
        "vgg11": models.vgg11,
        "vgg13": models.vgg13,
        "vgg16": models.vgg16,
        "vgg19": models.vgg19
    }
    weights_attr = getattr(models, model_name.upper() + "_Weights")
    model = vgg_models[model_name](weights=weights_attr.IMAGENET1K_V1)
    for i, (name, param) in enumerate(model.features.named_parameters()):
        param.requires_grad = i >= freeze_until
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *list(model.classifier[:fc_layer_index])
    )
    return feature_extractor
