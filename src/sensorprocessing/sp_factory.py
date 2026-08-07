"""
sp_factory.py

Factory functions to create sensor processing objects based on an exp/run.

This version supports both single-view and multi-view sensor processors.
"""

from functools import partial

from sensorprocessing import (
    sp_conv_vae,
    sp_propriotuned_cnn,
    sp_aruco,
    sp_vit,
    sp_vit_multiview,
    sp_vit_concat_images,
    sp_propriotuned_cnn_multiview,
    sp_conv_vae_concat_multiview
)


def _cnn_exp_with_model(spexp, model=None):
    """Return ``spexp`` unchanged, or detached values with a model override."""
    if model is None:
        return spexp

    values = getattr(spexp, "values", spexp)
    if callable(values):  # ``dict.values`` is a method, unlike Experiment.values.
        values = spexp
    cnn_exp = values.copy()
    cnn_exp["model"] = model
    return cnn_exp


def _create_multiview_cnn(spexp, model=None):
    """Instantiate the generic multi-view CNN processor with an optional model."""
    return sp_propriotuned_cnn_multiview.MultiViewCNNSensorProcessing(
        _cnn_exp_with_model(spexp, model)
    )


def _create_singleview_cnn(spexp, model=None):
    """Instantiate the generic single-view CNN processor with an optional model."""
    return sp_propriotuned_cnn.ProprioTunedCNNSensorProcessing(
        _cnn_exp_with_model(spexp, model)
    )


_PROCESSOR_CLASSES = {
    "ConvVaeSensorProcessing": sp_conv_vae.ConvVaeSensorProcessing,
    "ConvVaeSensorProcessing_concat_multiview": (
        sp_conv_vae_concat_multiview.ConcatConvVaeSensorProcessing
    ),
    "VGG19ProprioTunedSensorProcessing": partial(
        _create_singleview_cnn, model="VGG19ProprioTunedRegression"
    ),
    "ResNetProprioTunedSensorProcessing": partial(
        _create_singleview_cnn, model="ResNetProprioTunedRegression"
    ),
    "ProprioTunedCNNSensorProcessing": _create_singleview_cnn,
    "ProprioTunedCNN": _create_singleview_cnn,
    "VGG19ProprioTunedSensorProcessing_multiview": partial(
        _create_multiview_cnn, model="MultiViewVGG19Model"
    ),
    "ResNetProprioTunedSensorProcessing_multiview": partial(
        _create_multiview_cnn, model="MultiViewResNetModel"
    ),
    "MultiViewCNNSensorProcessing": _create_multiview_cnn,
    "MultiViewCNN": _create_multiview_cnn,
    "Aruco": sp_aruco.ArucoSensorProcessing,
    "Vit": sp_vit.VitSensorProcessing,
    "Vit_multiview": sp_vit_multiview.MultiViewVitSensorProcessing,
    "Vit_concat_images": sp_vit_concat_images.ConcatImageVitSensorProcessing,
    "MultiViewVitSensorProcessing": sp_vit_multiview.MultiViewVitSensorProcessing,
}

_MULTIVIEW_CLASSES = {
    "ConvVaeSensorProcessing_concat_multiview",
    "ConvVaeSensorProcessing_multiview",
    "VGG19ProprioTunedSensorProcessing_multiview",
    "ResNetProprioTunedSensorProcessing_multiview",
    "MultiViewCNNSensorProcessing",
    "MultiViewCNN",
    "Vit_multiview",
    "Vit_concat_images",
    "MultiViewVitSensorProcessing",
}


def create_sp(spexp):
    """Gets the sensor processing component specified by the experiment.

    This factory function instantiates the appropriate sensor processing class
    based on the 'class' field in the experiment configuration.

    Args:
        spexp: Sensor processing experiment configuration dictionary

    Returns:
        Sensor processing object

    Raises:
        Exception: If the sensor processing class is unknown
    """
    sp_class = spexp.get("class", "")

    try:
        return _PROCESSOR_CLASSES[sp_class](spexp)
    except KeyError as error:
        available = "\n".join(f"  - {name}" for name in _PROCESSOR_CLASSES)
        raise Exception(
            f'Unknown sensor processing class: "{sp_class}"\n'
            f"Available classes:\n{available}"
        ) from error


def get_sp_class_name(sp):
    """Get a human-readable name for a sensor processor.

    Args:
        sp: Sensor processing object

    Returns:
        String with class name and relevant info
    """
    class_name = type(sp).__name__

    info_parts = [class_name]

    # Add multiview info
    if hasattr(sp, 'num_views'):
        info_parts.append(f"{sp.num_views} views")

    # Add latent size
    if hasattr(sp, 'latent_size'):
        info_parts.append(f"latent={sp.latent_size}")

    # Add fusion method for multiview
    if hasattr(sp, 'fusion_type'):
        info_parts.append(f"fusion={sp.fusion_type}")

    return " | ".join(info_parts)


def is_multiview_sp(spexp):
    """Check if an experiment config is for a multiview sensor processor.

    Args:
        spexp: Sensor processing experiment configuration

    Returns:
        Boolean indicating if this is a multiview config
    """
    sp_class = spexp.get("class", "")

    return sp_class in _MULTIVIEW_CLASSES or spexp.get("num_views", 1) > 1
