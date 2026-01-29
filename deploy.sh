#!/bin/bash
# Deploy news-pipeline to GCP Cloud Run Jobs with Cloud Scheduler
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - .env file in the same directory
#
# Usage:
#   ./deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables from .env
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo "Loading environment variables from .env..."
    # Export variables, ignoring comments and empty lines
    set -a
    source <(grep -v '^#' "${SCRIPT_DIR}/.env" | grep -v '^$' | grep '=')
    set +a
else
    echo "ERROR: .env file not found at ${SCRIPT_DIR}/.env"
    exit 1
fi

# Configuration
PROJECT_ID="praxis-db"
REGION="us-central1"
JOB_NAME="news-pipeline-job"
SCHEDULER_NAME="news-pipeline-schedule"
IMAGE="gcr.io/${PROJECT_ID}/news-pipeline"

echo "=== News Pipeline Deployment ==="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Ensure we're using the right project
gcloud config set project ${PROJECT_ID}

# Step 1: Build and push the container image using Cloud Build
echo "=== Building container image ==="
gcloud builds submit --tag ${IMAGE} .

# Step 2: Build environment variables string from .env
echo ""
echo "=== Preparing environment variables ==="

# List of env vars to include (exclude local paths and sensitive defaults)
ENV_VARS=""

add_env() {
    local name=$1
    local value="${!name}"
    if [ -n "$value" ]; then
        if [ -n "$ENV_VARS" ]; then
            ENV_VARS="${ENV_VARS},${name}=${value}"
        else
            ENV_VARS="${name}=${value}"
        fi
        echo "  ✓ ${name}"
    fi
}

# Add all required env vars
add_env "SUPABASE_DSN"
add_env "OPENAI_API_KEY"
add_env "OPENAI_EMBED_MODEL"
add_env "OPENROUTER_API_KEY"
add_env "OPENROUTER_MODEL"
add_env "NEWSDATA_API_KEY"
add_env "RECOMMENDER_API_KEY"
add_env "RECOMMENDER_API_URL"
add_env "GOOGLE_CLOUD_PROJECT"
add_env "PUBSUB_SUBSCRIPTION"
add_env "PUBSUB_TOPIC"
add_env "BATCH_SIZE"
add_env "MAX_CONCURRENT_BATCHES"
add_env "IDLE_TIMEOUT"
add_env "LOG_LEVEL"
add_env "INLINE_EMBEDDINGS"

# Step 3: Create or update the Cloud Run Job
echo ""
echo "=== Creating/updating Cloud Run Job ==="

# Check if job exists
if gcloud run jobs describe ${JOB_NAME} --region=${REGION} &>/dev/null; then
    echo "Updating existing job..."
    gcloud run jobs update ${JOB_NAME} \
        --region=${REGION} \
        --image=${IMAGE} \
        --memory=2Gi \
        --cpu=1 \
        --max-retries=1 \
        --task-timeout=30m \
        --set-env-vars="${ENV_VARS}"
else
    echo "Creating new job..."
    gcloud run jobs create ${JOB_NAME} \
        --region=${REGION} \
        --image=${IMAGE} \
        --memory=2Gi \
        --cpu=1 \
        --max-retries=1 \
        --task-timeout=30m \
        --set-env-vars="${ENV_VARS}"
fi

# Step 4: Set up Cloud Scheduler
echo ""
echo "=== Setting up Cloud Scheduler (every 3 hours) ==="

# Get the service account for Cloud Run
SERVICE_ACCOUNT=$(gcloud run jobs describe ${JOB_NAME} --region=${REGION} --format='value(spec.template.spec.serviceAccountName)')
if [ -z "$SERVICE_ACCOUNT" ]; then
    # Default compute service account
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')
    SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi

echo "Using service account: ${SERVICE_ACCOUNT}"

# Check if scheduler job exists
if gcloud scheduler jobs describe ${SCHEDULER_NAME} --location=${REGION} &>/dev/null; then
    echo "Updating existing scheduler..."
    gcloud scheduler jobs update http ${SCHEDULER_NAME} \
        --location=${REGION} \
        --schedule="0 */3 * * *" \
        --time-zone="UTC" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email=${SERVICE_ACCOUNT}
else
    echo "Creating new scheduler..."
    gcloud scheduler jobs create http ${SCHEDULER_NAME} \
        --location=${REGION} \
        --schedule="0 */3 * * *" \
        --time-zone="UTC" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email=${SERVICE_ACCOUNT}
fi

# Step 5: Grant Pub/Sub permissions
echo ""
echo "=== Granting Pub/Sub permissions ==="
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/pubsub.subscriber" \
    --condition=None \
    --quiet || true

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/pubsub.publisher" \
    --condition=None \
    --quiet || true

echo ""
echo "=== Deployment complete! ==="
echo ""
echo "Cloud Run Job: ${JOB_NAME}"
echo "Schedule: Every 3 hours (0 */3 * * * UTC)"
echo ""
echo "Test the job manually:"
echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION}"
echo ""
echo "View logs:"
echo "  gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION}"
echo ""
echo "Console:"
echo "  https://console.cloud.google.com/run/jobs/details/${REGION}/${JOB_NAME}?project=${PROJECT_ID}"
