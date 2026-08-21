from pathlib import Path


def test_batch_bootstrap_never_starts_ecs_synchronously_from_cloud_final() -> None:
    bootstrap = (
        Path(__file__).resolve().parents[1]
        / "infra"
        / "aws-map-builder"
        / "scripts"
        / "batch-bootstrap.mime"
    ).read_text(encoding="utf-8")

    assert "systemctl enable --now ecs" not in bootstrap
    assert "systemctl enable ecs" in bootstrap
    assert "systemctl start --no-block ecs" in bootstrap
