#!/bin/sh
set -eu

: "${MINIO_ROOT_USER:?MINIO_ROOT_USER is required}"
: "${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD is required}"
: "${MINIO_APP_BUCKET:?MINIO_APP_BUCKET is required}"
: "${MINIO_MILVUS_BUCKET:?MINIO_MILVUS_BUCKET is required}"
: "${MINIO_APP_ACCESS_KEY:?MINIO_APP_ACCESS_KEY is required}"
: "${MINIO_APP_SECRET_KEY:?MINIO_APP_SECRET_KEY is required}"
: "${MINIO_MILVUS_ACCESS_KEY:?MINIO_MILVUS_ACCESS_KEY is required}"
: "${MINIO_MILVUS_SECRET_KEY:?MINIO_MILVUS_SECRET_KEY is required}"

validate_bucket() {
    case "$1" in
        ""|*[!a-z0-9.-]*|.*|*.)
            echo "Invalid MinIO bucket name" >&2
            exit 1
            ;;
    esac
}

validate_access_key() {
    case "$1" in
        ""|*[!A-Za-z0-9._-]*)
            echo "Invalid MinIO access key" >&2
            exit 1
            ;;
    esac
}

validate_bucket "$MINIO_APP_BUCKET"
validate_bucket "$MINIO_MILVUS_BUCKET"
validate_access_key "$MINIO_APP_ACCESS_KEY"
validate_access_key "$MINIO_MILVUS_ACCESS_KEY"

if [ "${#MINIO_APP_SECRET_KEY}" -lt 16 ] || [ "${#MINIO_MILVUS_SECRET_KEY}" -lt 16 ]; then
    echo "MinIO service secrets must contain at least 16 characters" >&2
    exit 1
fi
if [ "$MINIO_APP_ACCESS_KEY" = "$MINIO_MILVUS_ACCESS_KEY" ]; then
    echo "MinIO application and Milvus identities must be distinct" >&2
    exit 1
fi

export MC_CONFIG_DIR=/tmp/.mc

endpoint="${MINIO_INIT_ENDPOINT:-http://minio:9000}"
attempt=1
until mc alias set local "$endpoint" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
    if [ "$attempt" -ge 60 ]; then
        echo "MinIO did not become available in time" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 2
done

mc mb --ignore-existing "local/${MINIO_APP_BUCKET}"
mc mb --ignore-existing "local/${MINIO_MILVUS_BUCKET}"
mc version enable "local/${MINIO_APP_BUCKET}"
mc version enable "local/${MINIO_MILVUS_BUCKET}"

write_bucket_policy() {
    bucket="$1"
    output="$2"
    printf '%s\n' "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetBucketLocation\",\"s3:GetBucketVersioning\",\"s3:ListBucket\",\"s3:ListBucketMultipartUploads\"],\"Resource\":[\"arn:aws:s3:::${bucket}\"]},{\"Effect\":\"Allow\",\"Action\":[\"s3:AbortMultipartUpload\",\"s3:DeleteObject\",\"s3:DeleteObjectVersion\",\"s3:GetObject\",\"s3:GetObjectVersion\",\"s3:ListMultipartUploadParts\",\"s3:PutObject\"],\"Resource\":[\"arn:aws:s3:::${bucket}/*\"]}]}" > "$output"
}

write_bucket_policy "$MINIO_APP_BUCKET" /tmp/knowledge-agent-app-policy.json
write_bucket_policy "$MINIO_MILVUS_BUCKET" /tmp/knowledge-agent-milvus-policy.json
mc admin policy create local knowledge-agent-app-v1 /tmp/knowledge-agent-app-policy.json
mc admin policy create local knowledge-agent-milvus-v1 /tmp/knowledge-agent-milvus-policy.json

if ! mc admin user info local "$MINIO_APP_ACCESS_KEY" >/dev/null 2>&1; then
    mc admin user add local "$MINIO_APP_ACCESS_KEY" "$MINIO_APP_SECRET_KEY"
fi
if ! mc admin user info local "$MINIO_MILVUS_ACCESS_KEY" >/dev/null 2>&1; then
    mc admin user add local "$MINIO_MILVUS_ACCESS_KEY" "$MINIO_MILVUS_SECRET_KEY"
fi
mc admin policy attach local knowledge-agent-app-v1 --user "$MINIO_APP_ACCESS_KEY"
mc admin policy attach local knowledge-agent-milvus-v1 --user "$MINIO_MILVUS_ACCESS_KEY"

mc stat "local/${MINIO_APP_BUCKET}" >/dev/null
mc stat "local/${MINIO_MILVUS_BUCKET}" >/dev/null
