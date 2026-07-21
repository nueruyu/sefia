"""`sefios` re-exports the core authoring surface as the same objects."""

import glyff
import sefia
import sefios


def test_reexports_are_the_source_objects():
    assert sefios.infer is sefia.infer
    assert sefios.preview is sefia.preview
    assert sefios.policy is sefia.policy
    assert sefios.profile is sefia.profile
    assert sefios.Policy is sefia.Policy
    assert sefios.Profile is sefia.Profile
    assert sefios.Tools is sefia.Tools
    assert sefios.AsRawText is sefia.AsRawText
    assert sefios.engrave is glyff.engrave


def test_authoring_surface_is_public():
    for name in (
        "infer",
        "preview",
        "policy",
        "profile",
        "Policy",
        "Profile",
        "Tools",
        "AsRawText",
        "engrave",
    ):
        assert name in sefios.__all__
