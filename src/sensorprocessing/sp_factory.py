"""
sp_factory.py

Factory functions to create sensor processing objects based on an exp/run.

This version supports both single-view and multi-view sensor processors.
"""

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

    # =========================================================================
    # CONV-VAE MODELS
    # =========================================================================

    if sp_class == "ConvVaeSensorProcessing":
        return sp_conv_vae.ConvVaeSensorProcessing(spexp)

    if sp_class == "ConvVaeSensorProcessing_concat_multiview":
        return sp_conv_vae_concat_multiview.ConcatConvVaeSensorProcessing(spexp)


    # =========================================================================
    # CNN MODELS (VGG, ResNet) - SINGLE VIEW
    # =========================================================================

    if sp_class == "VGG19ProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.VGG19ProprioTunedSensorProcessing(spexp)

    if sp_class == "ResNetProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.ResNetProprioTunedSensorProcessing(spexp)

    # =========================================================================
    # CNN MODELS (VGG, ResNet) - MULTI VIEW
    # =========================================================================

    if sp_class == "VGG19ProprioTunedSensorProcessing_multiview":
        return sp_propriotuned_cnn_multiview.MultiViewVGG19SensorProcessing(spexp)

    if sp_class == "ResNetProprioTunedSensorProcessing_multiview":
        return sp_propriotuned_cnn_multiview.MultiViewResNetSensorProcessing(spexp)

    # =========================================================================
    # ARUCO MARKER
    # =========================================================================

    if sp_class == "Aruco":
        return sp_aruco.ArucoSensorProcessing(spexp)

    # =========================================================================
    # VIT MODELS - SINGLE VIEW
    # =========================================================================

    if sp_class == "Vit":
        return sp_vit.VitSensorProcessing(spexp)

    # =========================================================================
    # VIT MODELS - MULTI VIEW
    # =========================================================================

    if sp_class == "Vit_multiview":
        return sp_vit_multiview.MultiViewVitSensorProcessing(spexp)

    if sp_class == "Vit_concat_images":
        return sp_vit_concat_images.ConcatImageVitSensorProcessing(spexp)

    # Also handle by name variations (for backwards compatibility)
    if sp_class == "MultiViewVitSensorProcessing":
        return sp_vit_multiview.MultiViewVitSensorProcessing(spexp)


    # =========================================================================
    # UNKNOWN CLASS
    # =========================================================================

    raise Exception(f'Unknown sensor processing class: "{sp_class}"\n'
                    f'Available classes:\n'
                    f'  - ConvVaeSensorProcessing\n'
                    f'  - ConvVaeSensorProcessing_concat_multiview\n'
                    f'  - ConvVaeSensorProcessing_multiview\n'
                    f'  - VGG19ProprioTunedSensorProcessing\n'
                    f'  - ResNetProprioTunedSensorProcessing\n'
                    f'  - VGG19ProprioTunedSensorProcessing_multiview\n'
                    f'  - ResNetProprioTunedSensorProcessing_multiview\n'
                    f'  - Aruco\n'
                    f'  - Vit\n'
                    f'  - Vit_multiview\n'
                    f'  - Vit_concat_images')


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

    multiview_classes = [
        "ConvVaeSensorProcessing_concat_multiview",
        "ConvVaeSensorProcessing_multiview",
        "VGG19ProprioTunedSensorProcessing_multiview",
        "ResNetProprioTunedSensorProcessing_multiview",
        "Vit_multiview",
        "Vit_concat_images",
        "MultiViewVitSensorProcessing"
    ]

    if sp_class in multiview_classes:
        return True

    # Also check num_views parameter
    if spexp.get("num_views", 1) > 1:
        return True

    return False