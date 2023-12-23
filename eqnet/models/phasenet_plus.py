from .phasenet import PhaseNet


def build_model(
    backbone="unet",
    log_scale=True,
    add_polarity=True,
    add_event=True,
    event_loss_weight=1.0,
    polarity_loss_weight=1.0,
    *args,
    **kwargs,
) -> PhaseNet:
    return PhaseNet(
        backbone=backbone,
        log_scale=log_scale,
        add_event=add_event,
        add_polarity=add_polarity,
        event_loss_weight=event_loss_weight,
        polarity_loss_weight=polarity_loss_weight,
    )
