import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(num_classes: int, freeze_until: int = 20):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Congelar capas hasta cierto punto
    if freeze_until is not None:
        for i, (name, param) in enumerate(model.features.named_parameters()):
            param.requires_grad = i >= freeze_until

    # Reemplazar la capa final
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
def build_efficientnet(model_name="efficientnet_b0", num_classes=10, freeze_until=20):
    model_name_v = model_name.lower()
    efficient_models = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_b4": models.efficientnet_b4,
        "efficientnet_b5": models.efficientnet_b5,
        "efficientnet_b6": models.efficientnet_b6,
        "efficientnet_b7": models.efficientnet_b7,
    }
    if model_name_v not in efficient_models:
        raise ValueError(f"Modelo {model_name} no soportado. Usa efficientnet_b0 a efficientnet_b7.")
    weights_attr = getattr(models, model_name + "_Weights")
    model = efficient_models[model_name_v](weights=weights_attr.IMAGENET1K_V1)

    # Cong
    if freeze_until is not None:
        for i, (name, param) in enumerate(model.features.named_parameters()):
            param.requires_grad = i >= freeze_until

    # Reemplazar la capa final de clasificación
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def build_efficientnet_extractor(model_name="efficientnet_b0", freeze_until: int = 20,flatten=True):
    weights_attr=getattr(models,model_name+"_Weights")
    model_fn=getattr(models,model_name.lower())
    model = model_fn(weights=weights_attr.IMAGENET1K_V1)

    # Congelar capas hasta cierto punto
    if freeze_until is not None:
        for i, (name, param) in enumerate(model.features.named_parameters()):
            param.requires_grad = i >= freeze_until
    feature_extractor=nn.Sequential(
        model.features,
        model.avgpool
    )
    if flatten:
        feature_extractor =nn.Sequential(
            feature_extractor,
            nn.Flatten()
        )
    return feature_extractor
1    # Reemplazar la capa final

def build_efficientnet_extractor_fc(model_name="efficientnet_b0", freeze_until=20, fc_layer_index=2, flatten=True):
    weights_attr = getattr(models, model_name + "_Weights")
    model_fn = getattr(models, model_name.lower())
    model = model_fn(weights=weights_attr.IMAGENET1K_V1)
    for i, (name, param) in enumerate(model.features.named_parameters()):
        param.requires_grad = i >= freeze_until
    # Extraer hasta cierta parte del clasificador
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *list(model.classifier[:fc_layer_index])
    )

    return feature_extractor
