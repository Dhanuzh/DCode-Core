//! Google Cloud Vertex AI model discovery.
//!
//! A single Vertex project serves both Gemini (`publishers/google`) and Claude
//! (`publishers/anthropic`), so one provider can expose everything the project
//! is entitled to instead of splitting them across platforms.
//!
//! Discovery deliberately does NOT trust the Model Garden catalog. That catalog
//! is a generic registry rather than a per-project entitlement list: probing a
//! real deployment showed it advertising `claude-sonnet-4-5` (which that project
//! cannot run) while omitting `claude-sonnet-4-6` (which it can). A per-model
//! GET returns 404 for entitled and non-entitled models alike, so it is not a
//! usable signal either. The catalog is therefore only a source of *candidates*,
//! unioned with a known-model seed list, and every candidate is validated with a
//! cheap live call before being surfaced.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dcode_api_types::{ModelInfo, VertexAuthMethod, VertexConfig};
use serde::{Deserialize, Serialize};

use crate::error::SystemError;

const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
/// Service-account assertions are short-lived; Google caps this at one hour.
const ASSERTION_TTL_SECS: u64 = 3600;

/// Models worth probing even when the catalog omits them. The catalog lags
/// real availability, so newer releases must be probed explicitly or they stay
/// invisible to users who actually have access.
const SEED_ANTHROPIC: &[&str] = &[
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-haiku-4-5",
];

const SEED_GOOGLE: &[&str] = &[
    "gemini-3.1-pro",
    "gemini-3.1-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
];

/// Vertex regional host. The `global` location is served by the unprefixed host.
pub fn host_for(location: &str) -> String {
    if location == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{location}-aiplatform.googleapis.com")
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn remote_error(err: &reqwest::Error) -> SystemError {
    SystemError::BadRequest(format!("Vertex request failed: {err}"))
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ServiceAccountKey {
    client_email: String,
    private_key: String,
    #[serde(default)]
    token_uri: Option<String>,
}

#[derive(Serialize)]
struct AssertionClaims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
}

/// Application Default Credentials, as written by
/// `gcloud auth application-default login`.
#[derive(Deserialize)]
struct AdcFile {
    client_id: String,
    client_secret: String,
    refresh_token: String,
}

/// Exchange a service-account key for an access token via a signed JWT
/// assertion (RFC 7523).
async fn token_from_service_account(client: &reqwest::Client, key_json: &str) -> Result<String, SystemError> {
    let key: ServiceAccountKey = serde_json::from_str(key_json)
        .map_err(|e| SystemError::BadRequest(format!("Invalid service account JSON: {e}")))?;
    let token_uri = key.token_uri.as_deref().unwrap_or(TOKEN_URL);

    let iat = now_secs();
    let claims = AssertionClaims {
        iss: &key.client_email,
        scope: SCOPE,
        aud: token_uri,
        iat,
        exp: iat + ASSERTION_TTL_SECS,
    };

    let encoding_key = jsonwebtoken::EncodingKey::from_rsa_pem(key.private_key.as_bytes())
        .map_err(|e| SystemError::BadRequest(format!("Invalid service account private key: {e}")))?;
    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
    let assertion = jsonwebtoken::encode(&header, &claims, &encoding_key)
        .map_err(|e| SystemError::BadRequest(format!("Failed to sign assertion: {e}")))?;

    let resp = client
        .post(token_uri)
        .form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", &assertion),
        ])
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(|e| remote_error(&e))?;

    if !resp.status().is_success() {
        return Err(SystemError::BadRequest(format!(
            "Service account token exchange failed ({})",
            resp.status()
        )));
    }
    let body: TokenResponse = resp.json().await.map_err(|e| remote_error(&e))?;
    Ok(body.access_token)
}

/// Refresh an access token from local ADC. Developer-machine convenience; a
/// shipped build should configure a service account instead.
async fn token_from_adc(client: &reqwest::Client) -> Result<String, SystemError> {
    let path = adc_path().ok_or_else(|| SystemError::BadRequest("Could not resolve the ADC path".into()))?;
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        SystemError::BadRequest(format!(
            "Could not read ADC at {}: {e}. Run `gcloud auth application-default login`.",
            path.display()
        ))
    })?;
    let adc: AdcFile =
        serde_json::from_str(&raw).map_err(|e| SystemError::BadRequest(format!("Invalid ADC file: {e}")))?;

    let resp = client
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", adc.client_id.as_str()),
            ("client_secret", adc.client_secret.as_str()),
            ("refresh_token", adc.refresh_token.as_str()),
        ])
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(|e| remote_error(&e))?;

    if !resp.status().is_success() {
        // Surface Google's own reason. `invalid_rapt` in particular means the
        // organisation enforces periodic reauthentication, which is a different
        // fix from a malformed or missing credential.
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        let hint = if detail.contains("invalid_rapt") {
            "Your organisation requires periodic reauthentication. "
        } else {
            ""
        };
        return Err(SystemError::BadRequest(format!(
            "ADC token refresh failed ({status}). {hint}Re-run `gcloud auth application-default login`. Details: {}",
            detail.chars().take(300).collect::<String>()
        )));
    }
    let body: TokenResponse = resp.json().await.map_err(|e| remote_error(&e))?;
    Ok(body.access_token)
}

fn adc_path() -> Option<std::path::PathBuf> {
    if let Ok(explicit) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        if !explicit.trim().is_empty() {
            return Some(std::path::PathBuf::from(explicit));
        }
    }
    // Windows gcloud writes under %APPDATA%; POSIX uses ~/.config.
    #[cfg(windows)]
    let base = dirs::config_dir();
    #[cfg(not(windows))]
    let base = dirs::home_dir().map(|h| h.join(".config"));

    Some(base?.join("gcloud").join("application_default_credentials.json"))
}

/// Mint a Google OAuth access token for a Vertex deployment.
///
/// Exposed so the agent layer can authenticate Gemini's OpenAI-compatible
/// Vertex endpoint, which takes a bearer token where other providers take a
/// static API key.
pub async fn mint_access_token(client: &reqwest::Client, cfg: &VertexConfig) -> Result<String, SystemError> {
    access_token(client, cfg).await
}

async fn access_token(client: &reqwest::Client, cfg: &VertexConfig) -> Result<String, SystemError> {
    match cfg.auth_method {
        VertexAuthMethod::ServiceAccount => {
            let key = cfg
                .service_account_json
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| SystemError::BadRequest("serviceAccountJson is required".into()))?;
            token_from_service_account(client, key).await
        }
        VertexAuthMethod::Adc => token_from_adc(client).await,
    }
}

// ---------------------------------------------------------------------------
// Catalog + probing
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PublisherModelsResponse {
    #[serde(default, rename = "publisherModels")]
    publisher_models: Vec<PublisherModel>,
}

#[derive(Deserialize)]
struct PublisherModel {
    #[serde(default)]
    name: String,
}

/// List candidate model ids for one publisher. Requires the `x-goog-user-project`
/// header, without which Vertex rejects ADC-derived tokens with 403.
async fn list_catalog(
    client: &reqwest::Client,
    token: &str,
    cfg: &VertexConfig,
    publisher: &str,
) -> Vec<String> {
    let url = format!(
        "https://{}/v1beta1/publishers/{publisher}/models?pageSize=200",
        host_for(&cfg.location)
    );
    let resp = client
        .get(&url)
        .bearer_auth(token)
        .header("x-goog-user-project", &cfg.project_id)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await;

    let Ok(resp) = resp else { return Vec::new() };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let Ok(body) = resp.json::<PublisherModelsResponse>().await else {
        return Vec::new();
    };

    body.publisher_models
        .into_iter()
        .filter_map(|m| m.name.rsplit('/').next().map(str::to_string))
        .filter(|id| !id.is_empty())
        .collect()
}

/// Cheap liveness check for one model. Gemini exposes `:countTokens`, which
/// costs nothing; Anthropic-on-Vertex has no equivalent, so it gets a
/// single-token `:rawPredict`.
async fn probe_model(
    client: &reqwest::Client,
    token: &str,
    cfg: &VertexConfig,
    publisher: &str,
    model: &str,
) -> bool {
    let host = host_for(&cfg.location);
    let base = format!(
        "https://{host}/v1/projects/{}/locations/{}/publishers/{publisher}/models/{model}",
        cfg.project_id, cfg.location
    );

    let (url, body) = if publisher == "anthropic" {
        (
            format!("{base}:rawPredict"),
            serde_json::json!({
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 1,
            }),
        )
    } else {
        (
            format!("{base}:countTokens"),
            serde_json::json!({
                "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
            }),
        )
    };

    client
        .post(&url)
        .bearer_auth(token)
        .header("x-goog-user-project", &cfg.project_id)
        .json(&body)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Discover every model the configured Vertex project can actually run,
/// across both the Gemini and Claude publishers.
pub(super) async fn fetch_vertex(
    client: &reqwest::Client,
    cfg: Option<&VertexConfig>,
) -> Result<Vec<ModelInfo>, SystemError> {
    let cfg = cfg.ok_or_else(|| SystemError::BadRequest("Vertex AI requires vertexConfig".into()))?;
    if cfg.project_id.trim().is_empty() {
        return Err(SystemError::BadRequest("projectId is required".into()));
    }
    if cfg.location.trim().is_empty() {
        return Err(SystemError::BadRequest("location is required".into()));
    }

    let token = access_token(client, cfg).await?;

    let mut found = Vec::new();
    for (publisher, seed) in [("google", SEED_GOOGLE), ("anthropic", SEED_ANTHROPIC)] {
        let mut candidates = list_catalog(client, &token, cfg, publisher).await;
        for extra in seed {
            if !candidates.iter().any(|c| c == extra) {
                candidates.push((*extra).to_string());
            }
        }

        // Probe concurrently — candidate lists reach a few dozen entries and
        // each probe is a round trip.
        let checks = candidates.into_iter().map(|id| {
            let client = client.clone();
            let token = token.clone();
            let cfg = cfg.clone();
            async move {
                let ok = probe_model(&client, &token, &cfg, publisher, &id).await;
                (id, ok)
            }
        });
        let results = futures_util::future::join_all(checks).await;
        found.extend(results.into_iter().filter(|(_, ok)| *ok).map(|(id, _)| ModelInfo::Id(id)));
    }

    if found.is_empty() {
        return Err(SystemError::BadRequest(format!(
            "No usable models found in project {} at location {}. Check the project, the location, and that models are enabled.",
            cfg.project_id, cfg.location
        )));
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_location_uses_unprefixed_host() {
        assert_eq!(host_for("global"), "aiplatform.googleapis.com");
    }

    #[test]
    fn regional_location_is_prefixed() {
        assert_eq!(host_for("us-east5"), "us-east5-aiplatform.googleapis.com");
    }

    #[tokio::test]
    async fn missing_config_is_rejected() {
        let client = reqwest::Client::new();
        let err = fetch_vertex(&client, None).await.unwrap_err();
        assert!(matches!(err, SystemError::BadRequest(_)));
    }

    #[tokio::test]
    async fn empty_project_id_is_rejected() {
        let client = reqwest::Client::new();
        let cfg = VertexConfig {
            auth_method: VertexAuthMethod::Adc,
            project_id: "  ".into(),
            location: "global".into(),
            service_account_json: None,
        };
        let err = fetch_vertex(&client, Some(&cfg)).await.unwrap_err();
        assert!(matches!(err, SystemError::BadRequest(_)));
    }

    #[tokio::test]
    async fn service_account_requires_key_json() {
        let client = reqwest::Client::new();
        let cfg = VertexConfig {
            auth_method: VertexAuthMethod::ServiceAccount,
            project_id: "proj".into(),
            location: "global".into(),
            service_account_json: None,
        };
        let err = fetch_vertex(&client, Some(&cfg)).await.unwrap_err();
        assert!(matches!(err, SystemError::BadRequest(_)));
    }
}
