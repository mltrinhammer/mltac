from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSet:
    """Definition of one modelling feature set.

    A feature set may be a single stream, such as Swin, or a fixed
    concatenation of streams, such as OpenFace2 + OpenFace3.
    """

    name: str
    streams: tuple[str, ...]
    description: str


# Feature-set registry.
#
# Each modelling experiment should refer to these stable names instead of
# spelling out stream names in multiple scripts. This keeps the definition of
# "visual_openface" or "audio_w2vbert2" consistent across preprocessing,
# transform fitting, and later model training.
FEATURE_SETS: dict[str, FeatureSet] = {
    "audio_egemaps": FeatureSet(
        name="audio_egemaps",
        streams=("audio.egemapsv2",),
        description="eGeMAPS acoustic descriptors.",
    ),
    "audio_w2vbert2": FeatureSet(
        name="audio_w2vbert2",
        streams=("audio.w2vbert2_embeddings",),
        description="W2V-BERT2 audio embeddings.",
    ),
    "visual_swin": FeatureSet(
        name="visual_swin",
        streams=("swin",),
        description="Swin visual embeddings.",
    ),
    "visual_openface": FeatureSet(
        name="visual_openface",
        streams=("openface2", "openface3"),
        description="Full OpenFace streams available in the corpus.",
    ),
    "visual_openpose": FeatureSet(
        name="visual_openpose",
        streams=("openpose",),
        description="Full OpenPose stream available in the corpus.",
    ),
}


def get_feature_set(name: str) -> FeatureSet:
    """Return a registered feature set with a helpful error for CLI use."""

    try:
        return FEATURE_SETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_SETS))
        raise KeyError(f"Unknown feature set {name!r}. Valid options: {valid}") from exc
