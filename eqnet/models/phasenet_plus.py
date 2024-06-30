from .phasenet import PhaseNet


def build_model(
    backbone="unet",
    init_features=16,
    upsample="interpolate",
    log_scale=True,
    add_polarity=True,
    add_event=True,
    event_center_loss_weight=1.0,
    event_time_loss_weight=1.0,
    polarity_loss_weight=0.2,
    *args,
    **kwargs,
) -> PhaseNet:
    return PhaseNet(
        backbone=backbone,
        upsample=upsample,
        init_features=init_features,
        log_scale=log_scale,
        add_event=add_event,
        add_polarity=add_polarity,
        event_center_loss_weight=event_center_loss_weight,
        event_time_loss_weight=event_time_loss_weight,
        polarity_loss_weight=polarity_loss_weight,
    )
