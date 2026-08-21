#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /var/log/fireviewer-map-builder-bootstrap.log | logger -t fireviewer-bootstrap -s 2>/dev/console) 2>&1

dnf install -y docker xfsprogs jq

root_source="$(findmnt -n -o SOURCE /)"
root_parent="$(lsblk -no PKNAME "${root_source}" | head -n 1)"
if [[ -z "${root_parent}" ]]; then
  root_parent="$(basename "${root_source}")"
fi

mapfile -t scratch_candidates < <(
  lsblk -dpno NAME,TYPE,MOUNTPOINT | awk -v root="/dev/${root_parent}" '$2 == "disk" && $1 != root && $3 == "" { print $1 }'
)

if [[ "${#scratch_candidates[@]}" -ne 1 ]]; then
  echo "Expected exactly one unmounted non-root scratch disk; found ${#scratch_candidates[@]}" >&2
  exit 20
fi

scratch_device="${scratch_candidates[0]}"
scratch_bytes="$(blockdev --getsize64 "${scratch_device}")"
if (( scratch_bytes < 80 * 1024 * 1024 * 1024 || scratch_bytes > 101 * 1024 * 1024 * 1024 )); then
  echo "Scratch disk size is outside the authorized 80-100 GiB range: ${scratch_bytes} bytes" >&2
  exit 21
fi

filesystem_type="$(blkid -o value -s TYPE "${scratch_device}" || true)"
if [[ -z "${filesystem_type}" ]]; then
  mkfs.xfs -f "${scratch_device}"
elif [[ "${filesystem_type}" != "xfs" ]]; then
  echo "Refusing to overwrite unexpected scratch filesystem: ${filesystem_type}" >&2
  exit 22
fi

mkdir -p /scratch
scratch_uuid="$(blkid -o value -s UUID "${scratch_device}")"
if ! grep -q "UUID=${scratch_uuid}" /etc/fstab; then
  printf 'UUID=%s /scratch xfs defaults,nofail,noatime 0 2\n' "${scratch_uuid}" >> /etc/fstab
fi
mount /scratch
chmod 1777 /scratch

mkdir -p /scratch/docker /scratch/tmp
cat >/etc/docker/daemon.json <<'JSON'
{
  "data-root": "/scratch/docker",
  "log-driver": "local",
  "log-opts": {
    "max-size": "20m",
    "max-file": "3"
  }
}
JSON

systemctl enable --now docker
systemctl enable --now amazon-ssm-agent

cat >/etc/profile.d/fireviewer-scratch.sh <<'PROFILE'
export TMPDIR=/scratch/tmp
export TMP=/scratch/tmp
export TEMP=/scratch/tmp
export XDG_CACHE_HOME=/scratch/cache
PROFILE

touch /scratch/.fireviewer-ready
echo "FireViewer Map Builder host ready: scratch=${scratch_device}, docker-root=/scratch/docker"
