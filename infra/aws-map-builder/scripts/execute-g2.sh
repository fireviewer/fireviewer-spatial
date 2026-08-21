#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 12 ]]; then
  echo "usage: execute-g2.sh WORK_BUCKET REQUEST_KEY BASELINE_KEY COMPARATOR_KEY BUILDS_BUCKET OUTPUT_PREFIX IMAGE_REF IMAGE_DIGEST GIT_COMMIT LAUNCH_TEMPLATE_ID LAUNCH_TEMPLATE_VERSION REGION" >&2
  exit 2
fi

work_bucket="$1"
request_key="$2"
baseline_key="$3"
comparator_key="$4"
builds_bucket="$5"
output_prefix="$6"
image_ref="$7"
image_digest="$8"
git_commit="$9"
launch_template_id="${10}"
launch_template_version="${11}"
region="${12}"

export AWS_DEFAULT_REGION="${region}"
export AWS_REGION="${region}"
export TMPDIR=/scratch/tmp
export TMP=/scratch/tmp
export TEMP=/scratch/tmp
export XDG_CACHE_HOME=/scratch/cache

test -f /scratch/.fireviewer-ready
if aws s3api head-object --bucket "${builds_bucket}" --key "${output_prefix}/zone.done.json" >/dev/null 2>&1; then
  echo "Refusing to overwrite an already completed build" >&2
  exit 30
fi

run_root=/scratch/fireviewer-g2
input_root="${run_root}/input"
builder_scratch="${run_root}/work"
output_root="${run_root}/output"
control_root="${run_root}/control"
mkdir -p "${input_root}" "${builder_scratch}" "${output_root}" "${control_root}"

aws s3 cp "s3://${work_bucket}/${request_key}" "${input_root}/request.json" --only-show-errors
aws s3 cp "s3://${work_bucket}/${baseline_key}" "${input_root}/semantic-baseline.json" --only-show-errors
aws s3 cp "s3://${work_bucket}/${comparator_key}" "${input_root}/compare-semantic-parity.py" --only-show-errors

registry="${image_ref%%/*}"
aws ecr get-login-password | docker login --username AWS --password-stdin "${registry}" >/dev/null
docker pull "${image_ref}"

observed_repo_digest="$(docker image inspect "${image_ref}" --format '{{index .RepoDigests 0}}')"
observed_digest="${observed_repo_digest##*@}"
observed_commit="$(docker image inspect "${image_ref}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
observed_platform="$(docker image inspect "${image_ref}" --format '{{.Os}}/{{.Architecture}}')"
if [[ "${observed_digest}" != "${image_digest}" || "${observed_commit}" != "${git_commit}" || "${observed_platform}" != "linux/amd64" ]]; then
  echo "Image identity mismatch: digest=${observed_digest}, commit=${observed_commit}, platform=${observed_platform}" >&2
  exit 31
fi

sample_file="${control_root}/host-samples.tsv"
stop_file="${control_root}/stop-monitor"
rm -f "${stop_file}"
(
  while [[ ! -f "${stop_file}" ]]; do
    read -r mem_total_kib mem_available_kib swap_total_kib swap_free_kib < <(
      awk '
        /^MemTotal:/ { mt=$2 }
        /^MemAvailable:/ { ma=$2 }
        /^SwapTotal:/ { st=$2 }
        /^SwapFree:/ { sf=$2 }
        END { print mt, ma, st, sf }
      ' /proc/meminfo
    )
    scratch_used_bytes="$(df -B1 --output=used /scratch | tail -n 1 | tr -d ' ')"
    printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$(( (mem_total_kib - mem_available_kib) * 1024 ))" "$(( (swap_total_kib - swap_free_kib) * 1024 ))" "${scratch_used_bytes}" >> "${sample_file}"
    sleep 1
  done
) &
monitor_pid=$!

stop_monitor() {
  touch "${stop_file}"
  wait "${monitor_pid}" 2>/dev/null || true
}
trap stop_monitor EXIT

build_started_epoch="$(date +%s)"
docker run --rm \
  --cpus 2 \
  --memory 7g \
  --memory-swap 7g \
  --mount "type=bind,src=${input_root},dst=/input,readonly" \
  --mount "type=bind,src=${builder_scratch},dst=/scratch" \
  --mount "type=bind,src=${output_root},dst=/output" \
  "${image_ref}" \
  --request /input/request.json \
  --scratch-root /scratch \
  --output /output
build_finished_epoch="$(date +%s)"

docker run --rm \
  --entrypoint python \
  --mount "type=bind,src=${input_root},dst=/input,readonly" \
  --mount "type=bind,src=${output_root},dst=/output" \
  "${image_ref}" \
  /input/compare-semantic-parity.py \
  --baseline /input/semantic-baseline.json \
  --output /output \
  --write /output/manifests/semantic-parity.json

stop_monitor
trap - EXIT

host_ram_peak_bytes="$(awk 'BEGIN { m=0 } $2>m { m=$2 } END { print m+0 }' "${sample_file}")"
host_swap_peak_bytes="$(awk 'BEGIN { m=0 } $3>m { m=$3 } END { print m+0 }' "${sample_file}")"
scratch_peak_bytes="$(awk 'BEGIN { m=0 } $4>m { m=$4 } END { print m+0 }' "${sample_file}")"
system_oom_events="$(journalctl -k --since "@${build_started_epoch}" --no-pager 2>/dev/null | grep -Eci 'out of memory|oom-kill|killed process' || true)"

metadata_token="$(curl -fsS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 300' http://169.254.169.254/latest/api/token)"
instance_id="$(curl -fsS -H "X-aws-ec2-metadata-token: ${metadata_token}" http://169.254.169.254/latest/meta-data/instance-id)"
ami_id="$(curl -fsS -H "X-aws-ec2-metadata-token: ${metadata_token}" http://169.254.169.254/latest/meta-data/ami-id)"
instance_type="$(curl -fsS -H "X-aws-ec2-metadata-token: ${metadata_token}" http://169.254.169.254/latest/meta-data/instance-type)"

mkdir -p "${output_root}/provenance" "${output_root}/metrics"
jq -n \
  --arg instance_id "${instance_id}" \
  --arg instance_type "${instance_type}" \
  --arg ami_id "${ami_id}" \
  --arg region "${region}" \
  --arg launch_template_id "${launch_template_id}" \
  --arg launch_template_version "${launch_template_version}" \
  --arg image_digest "${image_digest}" \
  --arg git_commit "${git_commit}" \
  --arg builder_contract "fireviewer.map-job.v1" \
  --arg profile_version "factual-v2" \
  '{
    schema: "fireviewer.aws-execution-provenance.v1",
    instance_id: $instance_id,
    instance_type: $instance_type,
    ami_id: $ami_id,
    region: $region,
    launch_template_id: $launch_template_id,
    launch_template_version: $launch_template_version,
    image_digest: $image_digest,
    git_commit: $git_commit,
    builder_contract: $builder_contract,
    profile_version: $profile_version,
    scratch: {type: "gp3", size_gib: 100, iops: 3000, throughput_mibps: 125, delete_on_termination: true}
  }' > "${output_root}/provenance/aws-execution.json"

hashes_path="${output_root}/manifests/hashes.json"
append_hash_entry() {
  local relative_path="$1"
  local local_path="${output_root}/${relative_path}"
  local byte_count sha256 temporary
  byte_count="$(stat -c %s "${local_path}")"
  sha256="$(sha256sum "${local_path}" | awk '{print $1}')"
  temporary="$(mktemp "${output_root}/manifests/.hashes.XXXXXX")"
  jq \
    --arg path "${relative_path}" \
    --arg sha256 "${sha256}" \
    --argjson byte_count "${byte_count}" \
    '.artifacts |= ([.[] | select(.path != $path)] + [{path: $path, byte_count: $byte_count, sha256: $sha256}] | sort_by(.path))' \
    "${hashes_path}" > "${temporary}"
  mv "${temporary}" "${hashes_path}"
}

append_hash_entry "manifests/semantic-parity.json"
append_hash_entry "provenance/aws-execution.json"

upload_started_epoch="$(date +%s)"
aws s3 cp \
  "${output_root}" \
  "s3://${builds_bucket}/${output_prefix}/" \
  --recursive \
  --exclude "zone.done.json" \
  --checksum-algorithm SHA256 \
  --only-show-errors
artifact_upload_finished_epoch="$(date +%s)"

jq -n \
  --argjson build_seconds "$((build_finished_epoch - build_started_epoch))" \
  --argjson artifact_upload_seconds "$((artifact_upload_finished_epoch - upload_started_epoch))" \
  --argjson host_ram_peak_bytes "${host_ram_peak_bytes}" \
  --argjson host_swap_peak_bytes "${host_swap_peak_bytes}" \
  --argjson scratch_peak_bytes "${scratch_peak_bytes}" \
  --argjson system_oom_events "${system_oom_events}" \
  '{
    schema: "fireviewer.aws-execution-metrics.v1",
    build_seconds: $build_seconds,
    artifact_upload_seconds: $artifact_upload_seconds,
    host_ram_peak_gb: ($host_ram_peak_bytes / 1073741824),
    host_swap_peak_gb: ($host_swap_peak_bytes / 1073741824),
    scratch_peak_gb_including_docker: ($scratch_peak_bytes / 1073741824),
    system_oom_events: $system_oom_events
  }' > "${output_root}/metrics/aws-execution-metrics.json"
append_hash_entry "metrics/aws-execution-metrics.json"

find "${output_root}" -type f ! -path "${hashes_path}" ! -path "${output_root}/zone.done.json" -printf '%P\n' | sort > "${control_root}/observed-files.txt"
jq -r '.artifacts[] | select(.path != "zone.done.json") | .path' "${hashes_path}" | sort > "${control_root}/sealed-files.txt"
if ! diff -u "${control_root}/sealed-files.txt" "${control_root}/observed-files.txt"; then
  echo "Output inventory differs from the sealed hashes manifest" >&2
  exit 42
fi

# These two files are finalized only after the bulk upload duration is known.
aws s3 cp \
  "${output_root}/metrics/aws-execution-metrics.json" \
  "s3://${builds_bucket}/${output_prefix}/metrics/aws-execution-metrics.json" \
  --checksum-algorithm SHA256 \
  --only-show-errors
aws s3 cp \
  "${hashes_path}" \
  "s3://${builds_bucket}/${output_prefix}/manifests/hashes.json" \
  --checksum-algorithm SHA256 \
  --only-show-errors

verify_critical_size() {
  local relative_path="$1"
  local expected_bytes observed_bytes
  if [[ "${relative_path}" == "manifests/hashes.json" ]]; then
    expected_bytes="$(stat -c %s "${hashes_path}")"
  else
    expected_bytes="$(jq -r --arg path "${relative_path}" '.artifacts[] | select(.path == $path) | .byte_count' "${hashes_path}")"
  fi
  observed_bytes="$(aws s3api head-object --bucket "${builds_bucket}" --key "${output_prefix}/${relative_path}" --query ContentLength --output text)"
  if [[ -z "${expected_bytes}" || "${observed_bytes}" != "${expected_bytes}" ]]; then
    echo "Critical S3 object size mismatch: ${relative_path} (${observed_bytes} != ${expected_bytes})" >&2
    exit 41
  fi
}

for critical_path in \
  "runtime/viewer-tiled/catalog.json" \
  "runtime/viewer-tiled/viewer-tiled-scene.v1.json" \
  "runtime/viewer-tiled/far.glb" \
  "scientific/zone.usda" \
  "scientific/zone.blend" \
  "manifests/hashes.json" \
  "metrics/build-metrics.json" \
  "metrics/aws-execution-metrics.json" \
  "provenance/aws-execution.json"; do
  verify_critical_size "${critical_path}"
done

artifact_count="$(jq '.artifacts | length' "${hashes_path}")"
uploaded_before_done="$(aws s3api list-objects-v2 --bucket "${builds_bucket}" --prefix "${output_prefix}/" --query 'length(Contents)' --output text)"
if [[ "${uploaded_before_done}" != "${artifact_count}" ]]; then
  echo "Unexpected S3 object count before zone.done.json: ${uploaded_before_done} != ${artifact_count}" >&2
  exit 43
fi

done_entry="$(jq -c '.artifacts[] | select(.path == "zone.done.json" and .publication_order == "last")' "${hashes_path}")"
if [[ -z "${done_entry}" ]]; then
  echo "zone.done.json is not marked for last publication" >&2
  exit 44
fi
done_expected_bytes="$(jq -r '.byte_count' <<<"${done_entry}")"
done_expected_hex="$(jq -r '.sha256' <<<"${done_entry}")"
done_observed_hex="$(sha256sum "${output_root}/zone.done.json" | awk '{print $1}')"
if [[ "${done_observed_hex}" != "${done_expected_hex}" || "$(stat -c %s "${output_root}/zone.done.json")" != "${done_expected_bytes}" ]]; then
  echo "Local zone.done.json differs from the sealed hashes manifest" >&2
  exit 45
fi
done_expected_b64="$(openssl dgst -sha256 -binary "${output_root}/zone.done.json" | openssl base64 -A)"
aws s3api put-object \
  --bucket "${builds_bucket}" \
  --key "${output_prefix}/zone.done.json" \
  --body "${output_root}/zone.done.json" \
  --checksum-sha256 "${done_expected_b64}" \
  --metadata "sha256=${done_expected_hex}" >/dev/null
done_head="$(aws s3api head-object --checksum-mode ENABLED --bucket "${builds_bucket}" --key "${output_prefix}/zone.done.json")"
if [[ "$(jq -r '.ContentLength' <<<"${done_head}")" != "${done_expected_bytes}" || "$(jq -r '.ChecksumSHA256' <<<"${done_head}")" != "${done_expected_b64}" || "$(jq -r '.Metadata.sha256' <<<"${done_head}")" != "${done_expected_hex}" ]]; then
  echo "Final zone.done.json verification failed" >&2
  exit 46
fi

uploaded_after_done="$(aws s3api list-objects-v2 --bucket "${builds_bucket}" --prefix "${output_prefix}/" --query 'length(Contents)' --output text)"
if [[ "${uploaded_after_done}" != "$((artifact_count + 1))" ]]; then
  echo "Unexpected S3 object count after zone.done.json: ${uploaded_after_done} != $((artifact_count + 1))" >&2
  exit 47
fi

echo "FIREVIEWER_AWS_G2_RESULT status=PASS instance_id=${instance_id} objects=${uploaded_after_done} prefix=s3://${builds_bucket}/${output_prefix}"
