"""
sp_factory.py

Factory functions to create sensor processing objects based on an exp/run
"""

from sensorprocessing import sp_conv_vae, sp_propriotuned_cnn, sp_aruco, sp_vit, sp_vit_multiview, sp_vit_concat_images, sp_propriotuned_cnn_multiview, sp_conv_vae_concat_multiview

def create_sp(spexp, device):
    """Gets the sensor processing component specified by the
    visual_proprioception experiment."""
    # spexp = Config().get_experiment(exp['sp_experiment'], exp['sp_run'])
    if spexp["class"] == "ConvVaeSensorProcessing":
        return sp_conv_vae.ConvVaeSensorProcessing(spexp, device)
    if exp["class"] == "ConvVaeSensorProcessing_concat_multiview":
        return sp_conv_vae_concat_multiview.ConcatConvVaeSensorProcessing(spexp, device)
    if exp["class"] == "ConvVaeSensorProcessing_multiview":
        return sp_conv_vae_multiview.ConvVaeSensorProcessing_multiview(spexp, device)
    if spexp['class']=="VGG19ProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.VGG19ProprioTunedSensorProcessing(spexp, device)
    if spexp['class']=="ResNetProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.ResNetProprioTunedSensorProcessing(spexp, device)
    if exp['class']=="VGG19ProprioTunedSensorProcessing_multiview":
        return sp_propriotuned_cnn_multiview.MultiViewVGG19SensorProcessing(spexp, device)
    if exp['class']=="ResNetProprioTunedSensorProcessing_multiview":
        return sp_propriotuned_cnn_multiview.MultiViewResNetSensorProcessing(spexp, device)
    if spexp['class']=="Aruco":
        return sp_aruco.ArucoSensorProcessing(spexp, device)
    if exp['class']=="Vit":
        return sp_vit.VitSensorProcessing(spexp, device)
    if exp['class'] == "Vit_multiview":
        return sp_vit_multiview.MultiViewVitSensorProcessing(spexp, device)
    if exp['class'] == "Vit_concat_images":
        return sp_vit_concat_images.ConcatImageVitSensorProcessing(spexp, device)
    raise Exception('Unknown sensor processing {exp["class"]}')


